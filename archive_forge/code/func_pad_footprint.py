import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def pad_footprint(footprint, *, pad_end=True):
    """Pad the footprint to an odd size along each dimension.

    Parameters
    ----------
    footprint : ndarray or tuple
        The input footprint or sequence of footprints
    pad_end : bool, optional
        If ``True``, pads at the end of each dimension (right side), otherwise
        pads on the front (left side).

    Returns
    -------
    padded : ndarray or tuple
        The footprint, padded to an odd size along each dimension.

    Examples
    --------
    >>> footprint = np.array([[0, 0],
    ...                       [1, 1],
    ...                       [1, 1]], np.uint8)
    >>> pad_footprint(footprint)
    array([[0, 0, 0],
           [1, 1, 0],
           [1, 1, 0]], dtype=uint8)

    """
    if _footprint_is_sequence(footprint):
        return tuple(((pad_footprint(fp, pad_end=pad_end), n) for fp, n in footprint))
    footprint = np.asarray(footprint)
    padding = []
    for sz in footprint.shape:
        padding.append(((0, 1) if pad_end else (1, 0)) if sz % 2 == 0 else (0, 0))
    return np.pad(footprint, padding)