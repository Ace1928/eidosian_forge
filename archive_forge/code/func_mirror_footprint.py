import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def mirror_footprint(footprint):
    """Mirror each dimension in the footprint.

    Parameters
    ----------
    footprint : ndarray or tuple
        The input footprint or sequence of footprints

    Returns
    -------
    inverted : ndarray or tuple
        The footprint, mirrored along each dimension.

    Examples
    --------
    >>> footprint = np.array([[0, 0, 0],
    ...                       [0, 1, 1],
    ...                       [0, 1, 1]], np.uint8)
    >>> mirror_footprint(footprint)
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]], dtype=uint8)

    """
    if _footprint_is_sequence(footprint):
        return tuple(((mirror_footprint(fp), n) for fp, n in footprint))
    footprint = np.asarray(footprint)
    return footprint[(slice(None, None, -1),) * footprint.ndim]