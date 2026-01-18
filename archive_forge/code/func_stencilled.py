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
@ngjit
def stencilled(arr, mask, out):
    M, N = arr.shape
    for y in range(M):
        for x in range(N):
            el = arr[y, x]
            for i in range(mask_size):
                for j in range(mask_size):
                    if mask[i, j]:
                        if np.isnan(el):
                            result = out[i + y, j + x]
                        elif np.isnan(out[i + y, j + x]):
                            result = el
                        else:
                            result = op(el, out[i + y, j + x])
                        out[i + y, j + x] = result