import scipy.linalg._interpolative as _id
import numpy as np
def iddr_aidi(m, n, k):
    """
    Initialize array for :func:`iddr_aid`.

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
        Initialization array to be used by :func:`iddr_aid`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.iddr_aidi(m, n, k)