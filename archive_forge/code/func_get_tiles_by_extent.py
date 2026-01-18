from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def get_tiles_by_extent(self, extent, level):
    xmin, ymin, xmax, ymax = extent
    txmin, tymax = self.meters_to_tile(xmin, ymin, level)
    txmax, tymin = self.meters_to_tile(xmax, ymax, level)
    tiles = []
    for ty in range(tymin, tymax + 1):
        for tx in range(txmin, txmax + 1):
            if self.is_valid_tile(tx, ty, level):
                t = (tx, ty, level, self.get_tile_meters(tx, ty, level))
                tiles.append(t)
    return tiles