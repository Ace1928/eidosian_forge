from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def get_level_by_extent(self, extent, height, width):
    x_rs = (extent[2] - extent[0]) / width
    y_rs = (extent[3] - extent[1]) / height
    resolution = max(x_rs, y_rs)
    i = 0
    for r in self._resolutions:
        if resolution > r:
            if i == 0:
                return 0
            if i > 0:
                return i - 1
        i += 1
    return i - 1