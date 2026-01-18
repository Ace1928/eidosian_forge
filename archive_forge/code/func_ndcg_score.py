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
@validate_params({'y_true': ['array-like'], 'y_score': ['array-like'], 'k': [Interval(Integral, 1, None, closed='left'), None], 'sample_weight': ['array-like', None], 'ignore_ties': ['boolean']}, prefer_skip_nested_validation=True)
def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False):
    """Compute Normalized Discounted Cumulative Gain.

    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount. Then divide by the best possible
    score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
    0 and 1.

    This ranking metric returns a high value if true labels are ranked high by
    ``y_score``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked. Negative values in `y_true` may result in an output
        that is not between 0 and 1.

    y_score : array-like of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).

    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.

    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.

    Returns
    -------
    normalized_discounted_cumulative_gain : float in [0., 1.]
        The averaged NDCG scores for all samples.

    See Also
    --------
    dcg_score : Discounted Cumulative Gain (not normalized).

    References
    ----------
    `Wikipedia entry for Discounted Cumulative Gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_

    Jarvelin, K., & Kekalainen, J. (2002).
    Cumulated gain-based evaluation of IR techniques. ACM Transactions on
    Information Systems (TOIS), 20(4), 422-446.

    Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T. Y. (2013, May).
    A theoretical analysis of NDCG ranking measures. In Proceedings of the 26th
    Annual Conference on Learning Theory (COLT 2013)

    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import ndcg_score
    >>> # we have groud-truth relevance of some answers to a query:
    >>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    >>> # we predict some scores (relevance) for the answers
    >>> scores = np.asarray([[.1, .2, .3, 4, 70]])
    >>> ndcg_score(true_relevance, scores)
    0.69...
    >>> scores = np.asarray([[.05, 1.1, 1., .5, .0]])
    >>> ndcg_score(true_relevance, scores)
    0.49...
    >>> # we can set k to truncate the sum; only top k answers contribute.
    >>> ndcg_score(true_relevance, scores, k=4)
    0.35...
    >>> # the normalization takes k into account so a perfect answer
    >>> # would still get 1.0
    >>> ndcg_score(true_relevance, true_relevance, k=4)
    1.0...
    >>> # now we have some ties in our prediction
    >>> scores = np.asarray([[1, 0, 0, 0, 1]])
    >>> # by default ties are averaged, so here we get the average (normalized)
    >>> # true relevance of our top predictions: (10 / 10 + 5 / 10) / 2 = .75
    >>> ndcg_score(true_relevance, scores, k=1)
    0.75...
    >>> # we can choose to ignore ties for faster results, but only
    >>> # if we know there aren't ties in our scores, otherwise we get
    >>> # wrong results:
    >>> ndcg_score(true_relevance,
    ...           scores, k=1, ignore_ties=True)
    0.5...
    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)
    if y_true.min() < 0:
        raise ValueError('ndcg_score should not be used on negative y_true values.')
    if y_true.ndim > 1 and y_true.shape[1] <= 1:
        raise ValueError(f'Computing NDCG is only meaningful when there is more than 1 document. Got {y_true.shape[1]} instead.')
    _check_dcg_target_type(y_true)
    gain = _ndcg_sample_scores(y_true, y_score, k=k, ignore_ties=ignore_ties)
    return np.average(gain, weights=sample_weight)