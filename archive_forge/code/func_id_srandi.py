import scipy.linalg._interpolative as _id
import numpy as np
def id_srandi(t):
    """
    Initialize seed values for :func:`id_srand` (any appropriately random
    numbers will do).

    :param t:
        Array of 55 seed values.
    :type t: :class:`numpy.ndarray`
    """
    t = np.asfortranarray(t)
    _id.id_srandi(t)