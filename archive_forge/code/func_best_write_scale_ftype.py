from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def best_write_scale_ftype(arr: np.ndarray, slope: npt.ArrayLike=1.0, inter: npt.ArrayLike=0.0, default: type[np.number]=np.float32) -> type[np.floating]:
    """Smallest float type to contain range of ``arr`` after scaling

    Scaling that will be applied to ``arr`` is ``(arr - inter) / slope``.

    Note that ``slope`` and ``inter`` get promoted to 1D arrays for this
    purpose to avoid the numpy scalar casting rules, which prevent scalars
    upcasting the array.

    Parameters
    ----------
    arr : array-like
        array that will be scaled
    slope : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``.
    inter : array-like, optional
        scalar such that output array will be ``(arr - inter) / slope``
    default : numpy type, optional
        minimum float type to return

    Returns
    -------
    ftype : numpy type
        Best floating point type for scaling.  If no floating point type
        prevents overflow, return the top floating point type.  If the input
        array ``arr`` already contains inf values, return the greater of the
        input type and the default type.

    Examples
    --------
    >>> arr = np.array([0, 1, 2], dtype=np.int16)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Specify higher default return value

    >>> best_write_scale_ftype(arr, 1, 0, default=np.float64) is np.float64
    True

    Even large values that don't overflow don't change output

    >>> arr = np.array([0, np.finfo(np.float32).max], dtype=np.float32)
    >>> best_write_scale_ftype(arr, 1, 0) is np.float32
    True

    Scaling > 1 reduces output values, so no upcast needed

    >>> best_write_scale_ftype(arr, np.float32(2), 0) is np.float32
    True

    Scaling < 1 increases values, so upcast may be needed (and is here)

    >>> best_write_scale_ftype(arr, np.float32(0.5), 0) is np.float64
    True
    """
    default = better_float_of(arr.dtype.type, default)
    if not np.all(np.isfinite(arr)):
        return default
    try:
        return _ftype4scaled_finite(arr, slope, inter, 'write', default)
    except ValueError:
        return OK_FLOATS[-1]