import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
@validate_params({'labels_true': ['array-like', None], 'labels_pred': ['array-like', None], 'contingency': ['array-like', 'sparse matrix', None]}, prefer_skip_nested_validation=True)
def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels
    of the same data. Where :math:`|U_i|` is the number of the samples
    in cluster :math:`U_i` and :math:`|V_j|` is the number of the
    samples in cluster :math:`V_j`, the Mutual Information
    between clusterings :math:`U` and :math:`V` is given as:

    .. math::

        MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}
        \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching :math:`U` (i.e
    ``label_true``) with :math:`V` (i.e. ``label_pred``) will return the
    same score value. This can be useful to measure the agreement of two
    independent label assignments strategies on the same dataset when the
    real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`U` in
        the above formula.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        A clustering of the data into disjoint subsets, called :math:`V` in
        the above formula.

    contingency : {array-like, sparse matrix} of shape             (n_classes_true, n_classes_pred), default=None
        A contingency matrix given by the
        :func:`~sklearn.metrics.cluster.contingency_matrix` function. If value
        is ``None``, it will be computed, otherwise the given value is used,
        with ``labels_true`` and ``labels_pred`` ignored.

    Returns
    -------
    mi : float
       Mutual information, a non-negative value, measured in nats using the
       natural logarithm.

    See Also
    --------
    adjusted_mutual_info_score : Adjusted against chance Mutual Information.
    normalized_mutual_info_score : Normalized Mutual Information.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    Examples
    --------
    >>> from sklearn.metrics import mutual_info_score
    >>> labels_true = [0, 1, 1, 0, 1, 0]
    >>> labels_pred = [0, 1, 0, 0, 1, 1]
    >>> mutual_info_score(labels_true, labels_pred)
    0.056...
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(contingency, accept_sparse=['csr', 'csc', 'coo'], dtype=[int, np.int32, np.int64])
    if isinstance(contingency, np.ndarray):
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        nzx, nzy, nz_val = sp.find(contingency)
    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    if pi.size == 1 or pj.size == 1:
        return 0.0
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(np.int64, copy=False)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = contingency_nm * (log_contingency_nm - log(contingency_sum)) + contingency_nm * log_outer
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)