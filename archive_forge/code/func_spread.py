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
def spread(img, px=1, shape='circle', how=None, mask=None, name=None):
    """Spread pixels in an image.

    Spreading expands each pixel a certain number of pixels on all sides
    according to a given shape, merging pixels using a specified compositing
    operator. This can be useful to make sparse plots more visible.

    Parameters
    ----------
    img : Image or other DataArray
    px : int, optional
        Number of pixels to spread on all sides
    shape : str, optional
        The shape to spread by. Options are 'circle' [default] or 'square'.
    how : str, optional
        The name of the compositing operator to use when combining
        pixels. Default of None uses 'over' operator for Image objects
        and 'add' operator otherwise.
    mask : ndarray, shape (M, M), optional
        The mask to spread over. If provided, this mask is used instead of
        generating one based on `px` and `shape`. Must be a square array
        with odd dimensions. Pixels are spread from the center of the mask to
        locations where the mask is True.
    name : string name, optional
        Optional string name to give to the Image object to return,
        to label results for display.
    """
    if not isinstance(img, xr.DataArray):
        raise TypeError('Expected `xr.DataArray`, got: `{0}`'.format(type(img)))
    is_image = isinstance(img, Image)
    name = img.name if name is None else name
    if mask is None:
        if not isinstance(px, int) or px < 0:
            raise ValueError('``px`` must be an integer >= 0')
        if px == 0:
            return img
        mask = _mask_lookup[shape](px)
    elif not (isinstance(mask, np.ndarray) and mask.ndim == 2 and (mask.shape[0] == mask.shape[1]) and (mask.shape[0] % 2 == 1)):
        raise ValueError('mask must be a square 2 dimensional ndarray with odd dimensions.')
        mask = mask if mask.dtype == 'bool' else mask.astype('bool')
    if how is None:
        how = 'over' if is_image else 'add'
    w = mask.shape[0]
    extra = w // 2
    M, N = img.shape[:2]
    padded_shape = (M + 2 * extra, N + 2 * extra)
    float_type = img.dtype in [np.float32, np.float64]
    fill_value = np.nan if float_type else 0
    if cupy and isinstance(img.data, cupy.ndarray):
        img.data = cupy.asnumpy(img.data)
    if is_image:
        kernel = _build_spread_kernel(how, is_image)
    elif float_type:
        kernel = _build_float_kernel(how, w)
    else:
        kernel = _build_int_kernel(how, w, img.dtype == np.uint32)

    def apply_kernel(layer):
        buf = np.full(padded_shape, fill_value, dtype=layer.dtype)
        kernel(layer.data, mask, buf)
        return buf[extra:-extra, extra:-extra].copy()
    if len(img.shape) == 2:
        out = apply_kernel(img)
    else:
        out = np.dstack([apply_kernel(img[:, :, category]) for category in range(img.shape[2])])
    return img.__class__(out, dims=img.dims, coords=img.coords, name=name)