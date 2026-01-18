from warnings import simplefilter
import numpy as np
from sklearn import naive_bayes
import wandb
import wandb.plots
from wandb.sklearn import calculate, utils
from . import shared
def feature_importances(model=None, feature_names=None, title='Feature Importance', max_num_features=50):
    """Log a plot depicting the relative importance of each feature for a classifier's decisions.

    Should only be called with a fitted classifer (otherwise an error is thrown).
    Only works with classifiers that have a feature_importances_ attribute, like trees.

    Arguments:
        model: (clf) Takes in a fitted classifier.
        feature_names: (list) Names for features. Makes plots easier to read by
                              replacing feature indexes with corresponding names.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_feature_importances(model, ["width", "height", "length"])
    ```
    """
    not_missing = utils.test_missing(model=model)
    correct_types = utils.test_types(model=model)
    model_fitted = utils.test_fitted(model)
    if not_missing and correct_types and model_fitted:
        feature_importance_chart = calculate.feature_importances(model, feature_names)
        wandb.log({'feature_importances': feature_importance_chart})