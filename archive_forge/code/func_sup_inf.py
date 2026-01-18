from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
def sup_inf(u):
    """SI operator."""
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError('u has an invalid number of dimensions (should be 2 or 3)')
    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i).astype(np.int8))
    return np.stack(erosions, axis=0).max(0)