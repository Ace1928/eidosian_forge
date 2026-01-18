import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def footprint_from_sequence(footprints):
    """Convert a footprint sequence into an equivalent ndarray.

    Parameters
    ----------
    footprints : tuple of 2-tuples
        A sequence of footprint tuples where the first element of each tuple
        is an array corresponding to a footprint and the second element is the
        number of times it is to be applied. Currently, all footprints should
        have odd size.

    Returns
    -------
    footprint : ndarray
        An single array equivalent to applying the sequence of ``footprints``.
    """
    shape = _shape_from_sequence(footprints)
    imag = np.zeros(shape, dtype=bool)
    imag[tuple((s // 2 for s in shape))] = 1
    return morphology.binary_dilation(imag, footprints)