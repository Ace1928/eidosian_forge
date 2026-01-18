from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def resample_2d(src, w, h, ds_method='mean', us_method='linear', fill_value=None, mode_rank=1, x_offset=(0, 0), y_offset=(0, 0), out=None):
    """
    Resample a 2-D grid to a new resolution.

    Parameters
    ----------
    src : np.ndarray
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    us_method : str (optional)
        Grid cell interpolation method for a possible upsampling
        (one of the *US_* constants, optional).
    fill_value : scalar (optional)
        If ``None``, it is taken from **src** if it is a masked array,
        otherwise from *out* if it is a masked array,
        otherwise numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, zwo
        means second most frequent value, and so forth.
    x_offset : tuple(float, float) (optional)
        Offsets for the x-axis indices in the source array (useful
        for distributed regridding where chunks are not aligned with
        the underlying array).
    y_offset : tuple(float, float) (optional)
        Offsets for the x-axis indices in the source array (useful
        for distributed regridding where chunks are not aligned with
        the underlying array).
    out : numpy.ndarray (optional)
        Alternate output array in which to place the result. The
        default is *None*; if provided, it must have the same shape as
        the expected output.

    Returns
    -------
    resampled : numpy.ndarray or dask.array.Array
        A resampled version of the *src* array.
    """
    out = _get_out(out, src, (h, w))
    if out is None:
        return src
    mask, use_mask = _get_mask(src)
    fill_value = _get_fill_value(fill_value, src, out)
    us_method = upsample_methods[us_method]
    ds_method = downsample_methods[ds_method]
    if isinstance(src, np.ma.MaskedArray):
        src = src.data
    resampled = _resample_2d(src, mask, use_mask, ds_method, us_method, fill_value, mode_rank, x_offset, y_offset, out)
    return _mask_or_not(resampled, src, fill_value)