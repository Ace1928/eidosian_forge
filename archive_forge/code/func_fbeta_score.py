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
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_pred': ['array-like', 'sparse matrix'], 'beta': [Interval(Real, 0.0, None, closed='both')], 'labels': ['array-like', None], 'pos_label': [Real, str, 'boolean', None], 'average': [StrOptions({'micro', 'macro', 'samples', 'weighted', 'binary'}), None], 'sample_weight': ['array-like', None], 'zero_division': [Options(Real, {0.0, 1.0}), 'nan', StrOptions({'warn'})]}, prefer_skip_nested_validation=True)
def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
    """Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter represents the ratio of recall importance to
    precision importance. `beta > 1` gives more weight to recall, while
    `beta < 1` favors precision. For example, `beta = 2` makes recall twice
    as important as precision, while `beta = 0.5` does the opposite.
    Asymptotically, `beta -> +inf` considers only recall, and `beta -> 0`
    only precision.

    The formula for F-beta score is:

    .. math::

       F_\\beta = \\frac{(1 + \\beta^2) \\text{tp}}
                        {(1 + \\beta^2) \\text{tp} + \\text{fp} + \\beta^2 \\text{fn}}

    Where :math:`\\text{tp}` is the number of true positives, :math:`\\text{fp}` is the
    number of false positives, and :math:`\\text{fn}` is the number of false negatives.

    Support beyond term:`binary` targets is achieved by treating :term:`multiclass`
    and :term:`multilabel` data as a collection of binary problems, one for each
    label. For the :term:`binary` case, setting `average='binary'` will return
    F-beta score for `pos_label`. If `average` is not `'binary'`, `pos_label` is
    ignored and F-beta score for both classes are computed, then averaged or both
    returned (when `average=None`). Similarly, for :term:`multiclass` and
    :term:`multilabel` targets, F-beta score for all `labels` are either returned or
    averaged depending on the `average` parameter. Use `labels` specify the set of
    labels to calculate F-beta score for.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float
        Determines the weight of recall in the combined score.

    labels : array-like, default=None
        The set of labels to include when `average != 'binary'`, and their
        order if `average is None`. Labels present in the data can be
        excluded, for example in multiclass classification to exclude a "negative
        class". Labels not present in the data can be included and will be
        "assigned" 0 samples. For multilabel targets, labels are column indices.
        By default, all labels in `y_true` and `y_pred` are used in sorted order.

        .. versionchanged:: 0.17
           Parameter `labels` improved for multiclass problem.

    pos_label : int, float, bool or str, default=1
        The class to report if `average='binary'` and the data is binary,
        otherwise this parameter is ignored.
        For multiclass or multilabel targets, set `labels=[pos_label]` and
        `average != 'binary'` to report metrics for one label only.

    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None,             default='binary'
        This parameter is required for multiclass/multilabel targets.
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
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative.

        Notes:
        - If set to "warn", this acts like 0, but a warning is also raised.
        - If set to `np.nan`, such values will be excluded from the average.

        .. versionadded:: 1.3
           `np.nan` option was added.

    Returns
    -------
    fbeta_score : float (if average is not None) or array of float, shape =        [n_unique_labels]
        F-beta score of the positive class in binary classification or weighted
        average of the F-beta score of each class for the multiclass task.

    See Also
    --------
    precision_recall_fscore_support : Compute the precision, recall, F-score,
        and support.
    multilabel_confusion_matrix : Compute a confusion matrix for each class or
        sample.

    Notes
    -----
    When ``true positive + false positive + false negative == 0``, f-score
    returns 0.0 and raises ``UndefinedMetricWarning``. This behavior can be
    modified by setting ``zero_division``.

    References
    ----------
    .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import fbeta_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    0.33...
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    array([0.71..., 0.        , 0.        ])
    >>> y_pred_empty = [0, 0, 0, 0, 0, 0]
    >>> fbeta_score(y_true, y_pred_empty,
    ...             average="macro", zero_division=np.nan, beta=0.5)
    0.12...
    """
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, average=average, warn_for=('f-score',), sample_weight=sample_weight, zero_division=zero_division)
    return f