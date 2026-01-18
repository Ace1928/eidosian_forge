from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def get_tile_meters(self, tx, ty, level):
    ty = invert_y_tile(ty, level)
    xmin, ymin = self.pixels_to_meters(tx * self.tile_size, ty * self.tile_size, level)
    xmax, ymax = self.pixels_to_meters((tx + 1) * self.tile_size, (ty + 1) * self.tile_size, level)
    return (xmin, ymin, xmax, ymax)