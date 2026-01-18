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
def apply_read_scaling(arr: np.ndarray, slope: Scalar | None=None, inter: Scalar | None=None) -> np.ndarray:
    """Apply scaling in `slope` and `inter` to array `arr`

    This is for loading the array from a file (as opposed to the reverse
    scaling when saving an array to file)

    Return data will be ``arr * slope + inter``. The trick is that we have to
    find a good precision to use for applying the scaling.  The heuristic is
    that the data is always upcast to the higher of the types from `arr,
    `slope`, `inter` if `slope` and / or `inter` are not default values. If the
    dtype of `arr` is an integer, then we assume the data more or less fills
    the integer range, and upcast to a type such that the min, max of
    ``arr.dtype`` * scale + inter, will be finite.

    Parameters
    ----------
    arr : array-like
    slope : None or float, optional
        slope value to apply to `arr` (``arr * slope + inter``).  None
        corresponds to a value of 1.0
    inter : None or float, optional
        intercept value to apply to `arr` (``arr * slope + inter``).  None
        corresponds to a value of 0.0

    Returns
    -------
    ret : array
        array with scaling applied.  Maybe upcast in order to give room for the
        scaling. If scaling is default (1, 0), then `ret` may be `arr` ``ret is
        arr``.
    """
    if slope is None:
        slope = 1.0
    if inter is None:
        inter = 0.0
    if (slope, inter) == (1, 0):
        return arr
    shape = arr.shape
    slope1d, inter1d = (np.atleast_1d(v) for v in (slope, inter))
    arr = np.atleast_1d(arr)
    if arr.dtype.kind in 'iu':
        default = slope1d.dtype.type if slope1d.dtype.kind == 'f' else np.float64
        ftype = int_scinter_ftype(arr.dtype, slope1d, inter1d, default)
        slope1d = slope1d.astype(ftype)
        inter1d = inter1d.astype(ftype)
    if slope1d != 1.0:
        arr = arr * slope1d
    if inter1d != 0.0:
        arr = arr + inter1d
    return arr.reshape(shape)