import json
import matplotlib.pyplot as plt
import argparse
import os
def plot_metrics(file_path):
    if not os.path.exists(file_path):
        print(f'File {file_path} does not exist.')
        return
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print('Invalid JSON file.')
            return
    directory = os.path.dirname(file_path)
    filename_prefix = os.path.basename(file_path).split('.')[0]
    plot_metric(data, 'loss', 'Epoch', 'Loss', 'Loss', ['b', 'r'])
    plt.savefig(os.path.join(directory, f'{filename_prefix}_train_and_validation_loss.png'))
    plt.close()
    plot_metric(data, 'perplexity', 'Epoch', 'Perplexity', 'Perplexity', ['g', 'm'])
    plt.savefig(os.path.join(directory, f'{filename_prefix}_train_and_validation_perplexity.png'))
    plt.close()
    plot_metrics_by_step(data, 'loss', 'Step', 'Loss', ['b', 'r'])
    plt.savefig(os.path.join(directory, f'{filename_prefix}_train_and_validation_loss_by_step.png'))
    plt.close()
    plot_metrics_by_step(data, 'perplexity', 'Step', 'Loss', ['g', 'm'])
    plt.savefig(os.path.join(directory, f'{filename_prefix}_train_and_validation_perplexity_by_step.png'))
    plt.close()