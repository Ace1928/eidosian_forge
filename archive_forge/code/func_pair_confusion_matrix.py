import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
@validate_params({'labels_true': ['array-like'], 'labels_pred': ['array-like']}, prefer_skip_nested_validation=True)
def pair_confusion_matrix(labels_true, labels_pred):
    """Pair confusion matrix arising from two clusterings [1]_.

    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings.

    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.

    Read more in the :ref:`User Guide <pair_confusion_matrix>`.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.

    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.

    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.

    See Also
    --------
    sklearn.metrics.rand_score : Rand Score.
    sklearn.metrics.adjusted_rand_score : Adjusted Rand Score.
    sklearn.metrics.adjusted_mutual_info_score : Adjusted Mutual Information.

    References
    ----------
    .. [1] :doi:`Hubert, L., Arabie, P. "Comparing partitions."
           Journal of Classification 2, 193–218 (1985).
           <10.1007/BF01908075>`

    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:

      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...

    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:

      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...

    Note that the matrix is not symmetric.
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = np.int64(labels_true.shape[0])
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True, dtype=np.int64)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C