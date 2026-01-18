import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import (
from ..utils._array_api import _union1d, _weighted_sum, get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _nanaverage
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _num_samples
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_pred': ['array-like', 'sparse matrix'], 'labels': ['array-like', None], 'pos_label': [Real, str, 'boolean', None], 'average': [StrOptions({'micro', 'macro', 'samples', 'weighted', 'binary'}), None], 'sample_weight': ['array-like', None], 'zero_division': [Options(Real, {0, 1}), StrOptions({'warn'})]}, prefer_skip_nested_validation=True)
def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
    """Jaccard similarity coefficient score.

    The Jaccard index [1], or Jaccard similarity coefficient, defined as
    the size of the intersection divided by the size of the union of two label
    sets, is used to compare set of predicted labels for a sample to the
    corresponding set of labels in ``y_true``.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return the
    Jaccard similarity coefficient for `pos_label`. If `average` is not `'binary'`,
    `pos_label` is ignored and scores for both classes are computed, then averaged or
    both returned (when `average=None`). Similarly, for :term:`multiclass` and
    :term:`multilabel` targets, scores for all `labels` are either returned or
    averaged depending on the `average` parameter. Use `labels` specify the set of
    labels to calculate the score for.

    Read more in the :ref:`User Guide <jaccard_similarity_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    labels : array-like of shape (n_classes,), default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.

    average : {'micro', 'macro', 'samples', 'weighted',             'binary'} or None, default='binary'
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", {0.0, 1.0}, default="warn"
        Sets the value to return when there is a zero division, i.e. when there
        there are no negative values in predictions and labels. If set to
        "warn", this acts like 0, but a warning is also raised.

    Returns
    -------
    score : float or ndarray of shape (n_unique_labels,), dtype=np.float64
        The Jaccard score. When `average` is not `None`, a single scalar is
        returned.

    See Also
    --------
    accuracy_score : Function for calculating the accuracy score.
    f1_score : Function for calculating the F1 score.
    multilabel_confusion_matrix : Function for computing a confusion matrix                                  for each class or sample.

    Notes
    -----
    :func:`jaccard_score` may be a poor metric if there are no
    positives for some samples or classes. Jaccard is undefined if there are
    no true or predicted labels, and our implementation will return a score
    of 0 with a warning.

    References
    ----------
    .. [1] `Wikipedia entry for the Jaccard index
           <https://en.wikipedia.org/wiki/Jaccard_index>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import jaccard_score
    >>> y_true = np.array([[0, 1, 1],
    ...                    [1, 1, 0]])
    >>> y_pred = np.array([[1, 1, 1],
    ...                    [1, 0, 0]])

    In the binary case:

    >>> jaccard_score(y_true[0], y_pred[0])
    0.6666...

    In the 2D comparison case (e.g. image similarity):

    >>> jaccard_score(y_true, y_pred, average="micro")
    0.6

    In the multilabel case:

    >>> jaccard_score(y_true, y_pred, average='samples')
    0.5833...
    >>> jaccard_score(y_true, y_pred, average='macro')
    0.6666...
    >>> jaccard_score(y_true, y_pred, average=None)
    array([0.5, 0.5, 1. ])

    In the multiclass case:

    >>> y_pred = [0, 2, 1, 2]
    >>> y_true = [0, 1, 2, 2]
    >>> jaccard_score(y_true, y_pred, average=None)
    array([1. , 0. , 0.33...])
    """
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
    samplewise = average == 'samples'
    MCM = multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels, samplewise=samplewise)
    numerator = MCM[:, 1, 1]
    denominator = MCM[:, 1, 1] + MCM[:, 0, 1] + MCM[:, 1, 0]
    if average == 'micro':
        numerator = np.array([numerator.sum()])
        denominator = np.array([denominator.sum()])
    jaccard = _prf_divide(numerator, denominator, 'jaccard', 'true or predicted', average, ('jaccard',), zero_division=zero_division)
    if average is None:
        return jaccard
    if average == 'weighted':
        weights = MCM[:, 1, 0] + MCM[:, 1, 1]
        if not np.any(weights):
            weights = None
    elif average == 'samples' and sample_weight is not None:
        weights = sample_weight
    else:
        weights = None
    return np.average(jaccard, weights=weights)