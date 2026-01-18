import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def get_data_window(arr, nodata=None):
    """Window covering the input array's valid data pixels.

    Parameters
    ----------
    arr: numpy ndarray, <= 3 dimensions
    nodata: number
        If None, will either return a full window if arr is not a masked
        array, or will use the mask to determine non-nodata pixels.
        If provided, it must be a number within the valid range of the
        dtype of the input array.

    Returns
    -------
    Window
    """
    if not 0 < arr.ndim <= 3:
        raise WindowError('get_data_window input array must have 1, 2, or 3 dimensions')
    if nodata is not None:
        if np.isnan(nodata):
            arr_mask = ~np.isnan(arr)
        else:
            arr_mask = arr != nodata
    elif np.ma.is_masked(arr):
        arr_mask = ~np.ma.getmask(arr)
    else:
        if arr.ndim == 1:
            full_window = ((0, arr.size), (0, 0))
        else:
            full_window = ((0, arr.shape[-2]), (0, arr.shape[-1]))
        return Window.from_slices(*full_window)
    if arr.ndim == 3:
        arr_mask = np.any(arr_mask, axis=0)
    v = []
    for nz in arr_mask.nonzero():
        if nz.size:
            v.append((nz.min(), nz.max() + 1))
        else:
            v.append((0, 0))
    if arr_mask.ndim == 1:
        v.append((0, 0))
    return Window.from_slices(*v)