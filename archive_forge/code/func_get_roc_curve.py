from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def get_roc_curve(model, data, thread_count=-1, plot=False):
    """
    Build points of ROC curve.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        A set of samples to build ROC curve with.

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    plot : bool, optional (default=False)
        If True, draw curve.

    Returns
    -------
    curve points : tuple of three arrays (fpr, tpr, thresholds)
    """
    if type(data) == Pool:
        data = [data]
    if not isinstance(data, list):
        raise CatBoostError('data must be a catboost.Pool or list of pools.')
    for pool in data:
        if not isinstance(pool, Pool):
            raise CatBoostError('one of data pools is not catboost.Pool')
    roc_curve = _get_roc_curve(model._object, data, thread_count)
    if plot:
        with _import_matplotlib() as plt:
            _draw(plt, roc_curve[0], roc_curve[1], 'False Positive Rate', 'True Positive Rate', 'ROC Curve')
    return roc_curve