import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def slice2len(slicer, in_len):
    """Output length after slicing original length `in_len` with `slicer`
    Parameters
    ----------
    slicer : slice object
    in_len : int

    Returns
    -------
    out_len : int
        Length after slicing

    Notes
    -----
    Returns same as ``len(np.arange(in_len)[slicer])``
    """
    if slicer == slice(None):
        return in_len
    full_slicer = fill_slicer(slicer, in_len)
    return _full_slicer_len(full_slicer)