import scipy.linalg._interpolative as _id
import numpy as np
def idd_reconint(idx, proj):
    """
    Reconstruct interpolation matrix from real ID.

    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Interpolation matrix.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idd_reconint(idx, proj)