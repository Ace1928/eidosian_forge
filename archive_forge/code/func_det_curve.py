import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import label_binarize
from ..utils import (
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import stable_cumsum
from ..utils.fixes import trapezoid
from ..utils.multiclass import type_of_target
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _check_sample_weight
from ._base import _average_binary_score, _average_multiclass_ovo_score
@validate_params({'y_true': ['array-like'], 'y_score': ['array-like'], 'pos_label': [Real, str, 'boolean', None], 'sample_weight': ['array-like', None]}, prefer_skip_nested_validation=True)
def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Compute error rates for different probability thresholds.

    .. note::
       This metric is used for evaluation of ranking and error tradeoffs of
       a binary classification task.

    Read more in the :ref:`User Guide <det_curve>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : ndarray of shape of (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rate (FPR) such that element i is the false positive
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false acceptance probability or fall-out.

    fnr : ndarray of shape (n_thresholds,)
        False negative rate (FNR) such that element i is the false negative
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false rejection or miss rate.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.

    See Also
    --------
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.
    DetCurveDisplay : DET curve visualization.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    precision_recall_curve : Compute precision-recall curve.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import det_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, fnr, thresholds = det_curve(y_true, y_scores)
    >>> fpr
    array([0.5, 0.5, 0. ])
    >>> fnr
    array([0. , 0.5, 0.5])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if len(np.unique(y_true)) != 2:
        raise ValueError('Only one class present in y_true. Detection error tradeoff curve is not defined in that case.')
    fns = tps[-1] - tps
    p_count = tps[-1]
    n_count = fps[-1]
    first_ind = fps.searchsorted(fps[0], side='right') - 1 if fps.searchsorted(fps[0], side='right') > 0 else None
    last_ind = tps.searchsorted(tps[-1]) + 1
    sl = slice(first_ind, last_ind)
    return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])