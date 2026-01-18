import warnings
from math import log
from numbers import Real
import numpy as np
from scipy import sparse as sp
from ...utils._param_validation import Interval, StrOptions, validate_params
from ...utils.multiclass import type_of_target
from ...utils.validation import check_array, check_consistent_length
from ._expected_mutual_info_fast import expected_mutual_information
@validate_params({'labels_true': ['array-like'], 'labels_pred': ['array-like'], 'average_method': [StrOptions({'arithmetic', 'max', 'min', 'geometric'})]}, prefer_skip_nested_validation=True)
def normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic'):
    """Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalization of the Mutual
    Information (MI) score to scale the results between 0 (no mutual
    information) and 1 (perfect correlation). In this function, mutual
    information is normalized by some generalized mean of ``H(labels_true)``
    and ``H(labels_pred))``, defined by the `average_method`.

    This measure is not adjusted for chance. Therefore
    :func:`adjusted_mutual_info_score` might be preferred.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Read more in the :ref:`User Guide <mutual_info_score>`.

    Parameters
    ----------
    labels_true : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    labels_pred : int array-like of shape (n_samples,)
        A clustering of the data into disjoint subsets.

    average_method : {'min', 'geometric', 'arithmetic', 'max'}, default='arithmetic'
        How to compute the normalizer in the denominator.

        .. versionadded:: 0.20

        .. versionchanged:: 0.22
           The default value of ``average_method`` changed from 'geometric' to
           'arithmetic'.

    Returns
    -------
    nmi : float
       Score between 0.0 and 1.0 in normalized nats (based on the natural
       logarithm). 1.0 stands for perfectly complete labeling.

    See Also
    --------
    v_measure_score : V-Measure (NMI with arithmetic mean option).
    adjusted_rand_score : Adjusted Rand Index.
    adjusted_mutual_info_score : Adjusted Mutual Information (adjusted
        against chance).

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      ... # doctest: +SKIP
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      ... # doctest: +SKIP
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      ... # doctest: +SKIP
      0.0
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    if classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0:
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    if mi == 0:
        return 0.0
    h_true, h_pred = (entropy(labels_true), entropy(labels_pred))
    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer