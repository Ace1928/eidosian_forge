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
def apply_kernel(layer):
    buf = np.full(padded_shape, fill_value, dtype=layer.dtype)
    kernel(layer.data, mask, buf)
    return buf[extra:-extra, extra:-extra].copy()