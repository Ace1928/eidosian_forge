from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def select_threshold(model=None, data=None, curve=None, FPR=None, FNR=None, thread_count=-1):
    """
    Selects a threshold for prediction.

    Parameters
    ----------
    model : catboost.CatBoost
        The trained model.

    data : catboost.Pool or list of catboost.Pool
        Set of samples to build ROC curve with.
        If set, curve parameter must not be set.

    curve : tuple of three arrays (fpr, tpr, thresholds)
        ROC curve points in format of get_roc_curve returned value.
        If set, data parameter must not be set.

    FPR : desired false-positive rate

    FNR : desired false-negative rate (only one of FPR and FNR should be chosen)

    thread_count : int (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    threshold : double
    """
    if data is not None:
        if curve is not None:
            raise CatBoostError('Only one of the parameters data and curve should be set.')
        if model is None:
            raise CatBoostError('model and data parameters should be set when curve parameter is None.')
        if type(data) == Pool:
            data = [data]
        if not isinstance(data, list):
            raise CatBoostError('data must be a catboost.Pool or list of pools.')
        for pool in data:
            if not isinstance(pool, Pool):
                raise CatBoostError('one of data pools is not catboost.Pool')
        return _select_threshold(model._object, data, None, FPR, FNR, thread_count)
    elif curve is not None:
        if not (isinstance(curve, list) or isinstance(curve, tuple)) or len(curve) != 3:
            raise CatBoostError('curve must be list or tuple of three arrays (fpr, tpr, thresholds).')
        return _select_threshold(None, None, curve, FPR, FNR, thread_count)
    else:
        raise CatBoostError('One of the parameters data and curve should be set.')