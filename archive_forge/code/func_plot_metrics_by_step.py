import json
import matplotlib.pyplot as plt
import argparse
import os
def plot_metrics_by_step(data, metric_name, x_label, y_label, colors):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_single_metric_by_step(data, f'train_step_{metric_name}', x_label, y_label, f'Train Step {metric_name.capitalize()}', colors[0])
    plt.subplot(1, 2, 2)
    plot_single_metric_by_step(data, f'val_step_{metric_name}', x_label, y_label, f'Validation Step {metric_name.capitalize()}', colors[1])
    plt.tight_layout()