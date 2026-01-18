import scipy.linalg._interpolative as _id
import numpy as np
def idd_frmi(m):
    """
    Initialize data for :func:`idd_frm`.

    :param m:
        Length of vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_frm`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idd_frmi(m)