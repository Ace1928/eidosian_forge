from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
from . import shared
def outlier_candidates(regressor=None, X=None, y=None):
    """Measures a datapoint's influence on regression model via cook's distance.

    Instances with high influences could potentially be outliers.

    Should only be called with a fitted regressor (otherwise an error is thrown).

    Please note this function fits the model on the training set when called.

    Arguments:
        model: (regressor) Takes in a fitted regressor.
        X: (arr) Training set features.
        y: (arr) Training set labels.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_outlier_candidates(model, X, y)
    ```
    """
    is_missing = utils.test_missing(regressor=regressor, X=X, y=y)
    correct_types = utils.test_types(regressor=regressor, X=X, y=y)
    is_fitted = utils.test_fitted(regressor)
    if is_missing and correct_types and is_fitted:
        y = np.asarray(y)
        outliers_chart = calculate.outlier_candidates(regressor, X, y)
        wandb.log({'outlier_candidates': outliers_chart})