# AI MRI Alzheimer's Detection

A deep learning pipeline that classifies brain MRI scans into dementia-severity stages. The notebook loads MRI images stored as raw bytes in Parquet files, runs exploratory data analysis on class balance and pixel brightness, trains a custom CNN from scratch, compares it against a fine-tuned ResNet18, and visualises predictions with Grad-CAM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wxseem-dev/AI-MRI-Alzheimer-s-Detection/blob/main/AI_MRI_Alzheimer's_Detection.ipynb)

## Features

- **Data Pipeline**: Loads MRI scans from Parquet files and decodes the raw image bytes into 128×128 grayscale pixel arrays
- **Exploratory Data Analysis**: Class distribution, class imbalance ratio, average brightness per class, and darkest/brightest scan outlier detection
- **Custom CNN (TinyCNN)**: A lightweight 3-block convolutional network trained from scratch for 4-class classification
- **Class-Imbalance Handling**: Inverse-frequency class weighting in the loss function for the severely underrepresented class
- **Hyperparameter Search**: Sweeps optimizer (Adam/SGD), learning rate, and batch size to find the best-performing configuration
- **Transfer Learning Baseline**: Fine-tunes a frozen, pretrained ResNet18 for comparison against the custom CNN
- **Model Interpretability**: Grad-CAM heatmaps showing which regions of each scan most influenced a prediction
- **Evaluation**: Confusion matrix and per-class precision/recall/F1 on a held-out test set
- **Reproducibility**: Fixed random seeds across NumPy, PyTorch, and CUDA

## Project Structure

```
AI-MRI-Alzheimer-s-Detection/
├── AI_MRI_Alzheimer's_Detection.ipynb   # Main notebook: data loading, EDA, training, evaluation
├── dataset/
│   ├── train.parquet                    # Training images (bytes) + labels
│   └── test.parquet                      # Test images (bytes) + labels
├── best_tinycnn_batch16.pth             # Saved weights for the best TinyCNN configuration
└── README.md                            # This file
```

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install pandas Pillow kagglehub ipywidgets numpy matplotlib pyarrow torch torchvision scikit-learn seaborn
```

## Getting the Data

The notebook expects `train.parquet` and `test.parquet` (given within the repository) inside a `dataset/` folder next to the notebook, with each row containing the raw MRI image bytes and an integer class label.

- Original data source: [NIAGADS](https://advp.niagads.org/downloads)

## Usage

Open the notebook in Google Colab (badge above), or run it locally with Jupyter:

```bash
jupyter notebook "AI_MRI_Alzheimer's_Detection.ipynb"
```

Run the cells in order:

1. Load the Parquet files and decode image bytes into grayscale pixel arrays
2. Explore class distribution, brightness statistics, and outlier scans
3. Train the TinyCNN baseline
4. Run the hyperparameter sweep and retrain with the best configuration
5. Evaluate the best model (confusion matrix, classification report, Grad-CAM)
6. Train and evaluate a ResNet18 transfer-learning baseline for comparison

## Model Architecture

### TinyCNN (trained from scratch)
- 3 convolutional blocks (1→16→32→64 channels), each followed by ReLU and 2×2 max pooling
- Fully connected head: `Flatten → Linear(16384, 128) → ReLU → Dropout(0.5) → Linear(128, 4)`
- Trained with class-weighted cross-entropy loss to counter class imbalance

### ResNet18 (transfer learning)
- ImageNet-pretrained backbone, all convolutional layers frozen
- Final fully connected layer replaced and fine-tuned for 4-class output
- Inputs resized to 224×224 and repeated to 3 channels to match ResNet's expected format

## Dataset at a Glance

- 5,120 training images / 1,280 test images, 128×128 grayscale
- 4 diagnostic classes, with one class heavily underrepresented (imbalance ratio of ~52:1 between the largest and smallest class)
- Train/validation split: 80% / 20%, stratified by class

## Key Results

| Model | Test Accuracy |
|---|---|
| TinyCNN — baseline config (Adam, lr=0.001, batch=32) | 0.57 |
| TinyCNN — best config (Adam, lr=0.001, batch=16) | **0.69** |
| ResNet18 — frozen backbone, fine-tuned head | 0.61 |

- The hyperparameter sweep showed batch size 16 with Adam clearly outperformed the other configurations
- The custom TinyCNN trained from scratch outperformed the ResNet18 transfer-learning baseline, likely because MRI scans differ quite a lot from the natural images ResNet18 was pretrained on
- The rarest class remained the hardest to classify due to having very few examples, even after class weighting
- Grad-CAM was used to visually inspect which regions of each scan the TinyCNN relied on for its predictions
