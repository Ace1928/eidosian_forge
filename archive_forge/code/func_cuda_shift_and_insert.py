from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit(device=True)
def cuda_shift_and_insert(target, value, index):
    """Insert a value into a 1D array at a particular index, but before doing
    that shift the previous values along one to make room. For use in
    ``FloatingNReduction`` classes such as ``max_n`` and ``first_n`` which
    store ``n`` values per pixel.

    Parameters
    ----------
    target : 1d numpy array
        Target pixel array.

    value : float
        Value to insert into target pixel array.

    index : int
        Index to insert at.

    Returns
    -------
    Index beyond insertion, i.e. where the first shifted value now sits.
    """
    n = len(target)
    for i in range(n - 1, index, -1):
        target[i] = target[i - 1]
    target[index] = value
    return index + 1