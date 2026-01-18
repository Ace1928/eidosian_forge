import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
def check_clusterings(labels_true, labels_pred):
    """Check that the labels arrays are 1D and of same dimension.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        The true labels.

    labels_pred : array-like of shape (n_samples,)
        The predicted labels.
    """
    labels_true = check_array(labels_true, ensure_2d=False, ensure_min_samples=0, dtype=None)
    labels_pred = check_array(labels_pred, ensure_2d=False, ensure_min_samples=0, dtype=None)
    type_label = type_of_target(labels_true)
    type_pred = type_of_target(labels_pred)
    if 'continuous' in (type_pred, type_label):
        msg = f'Clustering metrics expects discrete values but received {type_label} values for label, and {type_pred} values for target'
        warnings.warn(msg, UserWarning)
    if labels_true.ndim != 1:
        raise ValueError('labels_true must be 1D: shape is %r' % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError('labels_pred must be 1D: shape is %r' % (labels_pred.shape,))
    check_consistent_length(labels_true, labels_pred)
    return (labels_true, labels_pred)