import json
import matplotlib.pyplot as plt
import argparse
import os
def plot_metric(data, metric_name, x_label, y_label, title, colors):
    plt.figure(figsize=(7, 6))
    plt.plot(data[f'train_epoch_{metric_name}'], label=f'Train Epoch {metric_name.capitalize()}', color=colors[0])
    plt.plot(data[f'val_epoch_{metric_name}'], label=f'Validation Epoch {metric_name.capitalize()}', color=colors[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Train and Validation Epoch {title}')
    plt.legend()
    plt.tight_layout()