from warnings import simplefilter
import numpy as np
from sklearn import naive_bayes
import wandb
import wandb.plots
from wandb.sklearn import calculate, utils
from . import shared
def roc(y_true=None, y_probas=None, labels=None, plot_micro=True, plot_macro=True, classes_to_plot=None):
    """Log the receiver-operating characteristic curve.

    Arguments:
        y_true: (arr) Test set labels.
        y_probas: (arr) Test set predicted probabilities.
        labels: (list) Named labels for target variable (y). Makes plots easier to
                       read by replacing target values with corresponding index.
                       For example if `labels=['dog', 'cat', 'owl']` all 0s are
                       replaced by dog, 1s by cat.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_roc(y_true, y_probas, labels)
    ```
    """
    roc_chart = wandb.plots.roc.roc(y_true, y_probas, labels, plot_micro, plot_macro, classes_to_plot)
    wandb.log({'roc': roc_chart})