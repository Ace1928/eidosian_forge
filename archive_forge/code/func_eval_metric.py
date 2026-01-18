from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def eval_metric(label, approx, metric, weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs=None, thread_count=-1):
    """
    Evaluate metrics with raw approxes and labels.

    Parameters
    ----------
    label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
        Object labels with shape (n_objects,) or (n_object, n_target_dimension)

    approx : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
        Object approxes with shape (n_objects,) or (n_object, n_approx_dimension).

    metric : string
        Metric name.

    weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
        Object weights.

    group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
        Object group ids.

    group_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
        Group weights.

    subgroup_id : list or numpy.ndarray, optional (default=None)
        subgroup id for each instance.
        If not None, giving 1 dimensional array like data.

    pairs : list or numpy.ndarray or pandas.DataFrame or string or pathlib.Path
        The pairs description.
        If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
        The shape should be Nx2, where N is the pairs' count. The first element of the pair is
        the index of winner object in the training set. The second element of the pair is
        the index of loser object in the training set.
        If string or pathlib.Path, giving the path to the file with pairs description.

    thread_count : int, optional (default=-1)
        Number of threads to work with.
        If -1, then the number of threads is set to the number of CPU cores.

    Returns
    -------
    metric results : list with metric values.
    """
    if len(label) > 0:
        label = np.transpose(label) if isinstance(label[0], ARRAY_TYPES) else [label]
    if len(approx) == 0:
        approx = [[]]
    approx = np.transpose(approx) if isinstance(approx[0], ARRAY_TYPES) else [approx]
    return _eval_metric_util(label, approx, metric, weight, group_id, group_weight, subgroup_id, pairs, thread_count)