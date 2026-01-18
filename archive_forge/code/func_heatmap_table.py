import wandb
from wandb import util
from wandb.plots.utils import (
def heatmap_table(x_labels, y_labels, matrix_values, show_text):
    x_axis = []
    y_axis = []
    values = []
    count = 0
    for i, x in enumerate(x_labels):
        for j, y in enumerate(y_labels):
            x_axis.append(x)
            y_axis.append(y)
            values.append(matrix_values[j][i])
            count += 1
            if count >= chart_limit:
                wandb.termwarn('wandb uses only the first %d datapoints to create the plots.' % wandb.Table.MAX_ROWS)
                break
    if show_text:
        heatmap_key = 'wandb/heatmap/v1'
    else:
        heatmap_key = 'wandb/heatmap_no_text/v1'
    return wandb.visualize(heatmap_key, wandb.Table(columns=['x_axis', 'y_axis', 'values'], data=[[x_axis[i], y_axis[i], round(values[i], 2)] for i in range(len(x_axis))]))