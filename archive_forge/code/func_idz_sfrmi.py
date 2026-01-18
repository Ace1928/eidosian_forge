import scipy.linalg._interpolative as _id
import numpy as np
def idz_sfrmi(l, m):
    """
    Initialize data for :func:`idz_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idz_sfrmi(l, m)