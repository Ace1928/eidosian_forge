import scipy.linalg._interpolative as _id
import numpy as np
def idd_reconid(B, idx, proj):
    """
    Reconstruct matrix from real ID.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

    :return:
        Reconstructed matrix.
    :rtype: :class:`numpy.ndarray`
    """
    B = np.asfortranarray(B)
    if proj.size > 0:
        return _id.idd_reconid(B, idx, proj)
    else:
        return B[:, np.argsort(idx)]