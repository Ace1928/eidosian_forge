import scipy.linalg._interpolative as _id
import numpy as np
def idd_copycols(A, k, idx):
    """
    Reconstruct skeleton matrix from real ID.

    :param A:
        Original matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`

    :return:
        Skeleton matrix.
    :rtype: :class:`numpy.ndarray`
    """
    A = np.asfortranarray(A)
    return _id.idd_copycols(A, k, idx)