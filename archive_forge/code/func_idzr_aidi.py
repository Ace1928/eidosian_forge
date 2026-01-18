import scipy.linalg._interpolative as _id
import numpy as np
def idzr_aidi(m, n, k):
    """
    Initialize array for :func:`idzr_aid`.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Initialization array to be used by :func:`idzr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idzr_aidi(m, n, k)