from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def pixels_to_raster(self, px, py, level):
    map_size = self.tile_size << level
    return (px, map_size - py)