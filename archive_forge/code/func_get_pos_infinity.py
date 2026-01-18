from __future__ import annotations
import functools
from typing import Any
import numpy as np
from xarray.core import utils
def get_pos_infinity(dtype, max_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    max_for_int : bool
        Return np.iinfo(dtype).max instead of np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    if issubclass(dtype.type, np.floating):
        return np.inf
    if issubclass(dtype.type, np.integer):
        if max_for_int:
            return np.iinfo(dtype).max
        else:
            return np.inf
    if issubclass(dtype.type, np.complexfloating):
        return np.inf + 1j * np.inf
    return INF