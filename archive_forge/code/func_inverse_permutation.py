from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
from xarray.core.options import OPTIONS
def inverse_permutation(indices: np.ndarray, N: int | None=None) -> np.ndarray:
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.
    N : int, optional
        Size of the array

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
    if N is None:
        N = len(indices)
    inverse_permutation = np.full(N, -1, dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation