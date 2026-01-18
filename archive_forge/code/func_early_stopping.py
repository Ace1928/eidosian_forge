from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
def early_stopping(stopping_rounds: int, first_metric_only: bool=False, verbose: bool=True, min_delta: Union[float, List[float]]=0.0) -> _EarlyStoppingCallback:
    """Create a callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration`` attribute of a model.

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics.

        .. versionadded:: 4.0.0

    Returns
    -------
    callback : _EarlyStoppingCallback
        The callback that activates early stopping.
    """
    return _EarlyStoppingCallback(stopping_rounds=stopping_rounds, first_metric_only=first_metric_only, verbose=verbose, min_delta=min_delta)