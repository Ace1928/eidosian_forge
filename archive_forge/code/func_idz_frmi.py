import scipy.linalg._interpolative as _id
import numpy as np
def idz_frmi(m):
    """
    Initialize data for :func:`idz_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idz_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idz_frmi(m)