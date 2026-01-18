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
@validate_params({'y_true': ['array-like', 'sparse matrix'], 'y_pred': ['array-like', 'sparse matrix'], 'labels': ['array-like', None], 'sample_weight': ['array-like', None], 'raise_warning': ['boolean']}, prefer_skip_nested_validation=True)
def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None, raise_warning=True):
    """Compute binary classification positive and negative likelihood ratios.

    The positive likelihood ratio is `LR+ = sensitivity / (1 - specificity)`
    where the sensitivity or recall is the ratio `tp / (tp + fn)` and the
    specificity is `tn / (tn + fp)`. The negative likelihood ratio is `LR- = (1
    - sensitivity) / specificity`. Here `tp` is the number of true positives,
    `fp` the number of false positives, `tn` is the number of true negatives and
    `fn` the number of false negatives. Both class likelihood ratios can be used
    to obtain post-test probabilities given a pre-test probability.

    `LR+` ranges from 1 to infinity. A `LR+` of 1 indicates that the probability
    of predicting the positive class is the same for samples belonging to either
    class; therefore, the test is useless. The greater `LR+` is, the more a
    positive prediction is likely to be a true positive when compared with the
    pre-test probability. A value of `LR+` lower than 1 is invalid as it would
    indicate that the odds of a sample being a true positive decrease with
    respect to the pre-test odds.

    `LR-` ranges from 0 to 1. The closer it is to 0, the lower the probability
    of a given sample to be a false negative. A `LR-` of 1 means the test is
    useless because the odds of having the condition did not change after the
    test. A value of `LR-` greater than 1 invalidates the classifier as it
    indicates an increase in the odds of a sample belonging to the positive
    class after being classified as negative. This is the case when the
    classifier systematically predicts the opposite of the true label.

    A typical application in medicine is to identify the positive/negative class
    to the presence/absence of a disease, respectively; the classifier being a
    diagnostic test; the pre-test probability of an individual having the
    disease can be the prevalence of such disease (proportion of a particular
    population found to be affected by a medical condition); and the post-test
    probabilities would be the probability that the condition is truly present
    given a positive test result.

    Read more in the :ref:`User Guide <class_likelihood_ratios>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : array-like, default=None
        List of labels to index the matrix. This may be used to select the
        positive and negative classes with the ordering `labels=[negative_class,
        positive_class]`. If `None` is given, those that appear at least once in
        `y_true` or `y_pred` are used in sorted order.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    raise_warning : bool, default=True
        Whether or not a case-specific warning message is raised when there is a
        zero division. Even if the error is not raised, the function will return
        nan in such cases.

    Returns
    -------
    (positive_likelihood_ratio, negative_likelihood_ratio) : tuple
        A tuple of two float, the first containing the Positive likelihood ratio
        and the second the Negative likelihood ratio.

    Warns
    -----
    When `false positive == 0`, the positive likelihood ratio is undefined.
    When `true negative == 0`, the negative likelihood ratio is undefined.
    When `true positive + false negative == 0` both ratios are undefined.
    In such cases, `UserWarning` will be raised if raise_warning=True.

    References
    ----------
    .. [1] `Wikipedia entry for the Likelihood ratios in diagnostic testing
           <https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import class_likelihood_ratios
    >>> class_likelihood_ratios([0, 1, 0, 1, 0], [1, 1, 0, 0, 0])
    (1.5, 0.75)
    >>> y_true = np.array(["non-cat", "cat", "non-cat", "cat", "non-cat"])
    >>> y_pred = np.array(["cat", "cat", "non-cat", "non-cat", "non-cat"])
    >>> class_likelihood_ratios(y_true, y_pred)
    (1.33..., 0.66...)
    >>> y_true = np.array(["non-zebra", "zebra", "non-zebra", "zebra", "non-zebra"])
    >>> y_pred = np.array(["zebra", "zebra", "non-zebra", "non-zebra", "non-zebra"])
    >>> class_likelihood_ratios(y_true, y_pred)
    (1.5, 0.75)

    To avoid ambiguities, use the notation `labels=[negative_class,
    positive_class]`

    >>> y_true = np.array(["non-cat", "cat", "non-cat", "cat", "non-cat"])
    >>> y_pred = np.array(["cat", "cat", "non-cat", "non-cat", "non-cat"])
    >>> class_likelihood_ratios(y_true, y_pred, labels=["non-cat", "cat"])
    (1.5, 0.75)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type != 'binary':
        raise ValueError(f'class_likelihood_ratios only supports binary classification problems, got targets of type: {y_type}')
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels)
    if cm.shape == (1, 1):
        msg = 'samples of only one class were seen during testing '
        if raise_warning:
            warnings.warn(msg, UserWarning, stacklevel=2)
        positive_likelihood_ratio = np.nan
        negative_likelihood_ratio = np.nan
    else:
        tn, fp, fn, tp = cm.ravel()
        support_pos = tp + fn
        support_neg = tn + fp
        pos_num = tp * support_neg
        pos_denom = fp * support_pos
        neg_num = fn * support_neg
        neg_denom = tn * support_pos
        if support_pos == 0:
            msg = 'no samples of the positive class were present in the testing set '
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            positive_likelihood_ratio = np.nan
            negative_likelihood_ratio = np.nan
        if fp == 0:
            if tp == 0:
                msg = 'no samples predicted for the positive class'
            else:
                msg = 'positive_likelihood_ratio ill-defined and being set to nan '
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            positive_likelihood_ratio = np.nan
        else:
            positive_likelihood_ratio = pos_num / pos_denom
        if tn == 0:
            msg = 'negative_likelihood_ratio ill-defined and being set to nan '
            if raise_warning:
                warnings.warn(msg, UserWarning, stacklevel=2)
            negative_likelihood_ratio = np.nan
        else:
            negative_likelihood_ratio = neg_num / neg_denom
    return (positive_likelihood_ratio, negative_likelihood_ratio)