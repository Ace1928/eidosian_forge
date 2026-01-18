from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
def learning_curve(model=None, X=None, y=None, cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None):
    """Logs a plot depicting model performance against dataset size.

    Please note this function fits the model to datasets of varying sizes when called.

    Arguments:
        model: (clf or reg) Takes in a fitted regressor or classifier.
        X: (arr) Dataset features.
        y: (arr) Dataset labels.

    For details on the other keyword arguments, see the documentation for
    `sklearn.model_selection.learning_curve`.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_learning_curve(model, X, y)
    ```
    """
    not_missing = utils.test_missing(model=model, X=X, y=y)
    correct_types = utils.test_types(model=model, X=X, y=y)
    if not_missing and correct_types:
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        y = np.asarray(y)
        learning_curve_chart = calculate.learning_curve(model, X, y, cv, shuffle, random_state, train_sizes, n_jobs, scoring)
        wandb.log({'learning_curve': learning_curve_chart})