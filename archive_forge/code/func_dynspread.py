from __future__ import annotations
from collections.abc import Iterator
from io import BytesIO
import warnings
import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray
from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit
def dynspread(img, threshold=0.5, max_px=3, shape='circle', how=None, name=None):
    """Spread pixels in an image dynamically based on the image density.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible. Dynamic
    spreading determines how many pixels to spread based on a density
    heuristic.  Spreading starts at 1 pixel, and stops when the fraction
    of adjacent non-empty pixels reaches the specified threshold, or
    the max_px is reached, whichever comes first.

    Parameters
    ----------
    img : Image
    threshold : float, optional
        A tuning parameter in [0, 1], with higher values giving more
        spreading.
    max_px : int, optional
        Maximum number of pixels to spread on all sides.
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining
        pixels. Default of None uses 'over' operator for Image objects
        and 'add' operator otherwise.
    """
    is_image = isinstance(img, Image)
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    if not isinstance(max_px, int) or max_px < 0:
        raise ValueError('max_px must be >= 0')
    float_type = img.dtype in [np.float32, np.float64]
    if cupy and isinstance(img.data, cupy.ndarray):
        img.data = cupy.asnumpy(img.data)
    px_ = 0
    for px in range(1, max_px + 1):
        px_ = px
        if is_image:
            density = _rgb_density(img.data, px * 2)
        elif len(img.shape) == 2:
            density = _array_density(img.data, float_type, px * 2)
        else:
            masked = np.logical_not(np.isnan(img)) if float_type else img != 0
            flat_mask = np.sum(masked, axis=2, dtype='uint32')
            density = _array_density(flat_mask.data, False, px * 2)
        if density > threshold:
            px_ = px_ - 1
            break
    if px_ >= 1:
        return spread(img, px_, shape=shape, how=how, name=name)
    else:
        return img