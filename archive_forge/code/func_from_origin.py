from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
def from_origin(west, north, xsize, ysize):
    """Return an Affine transformation given upper left and pixel sizes.

    Return an Affine transformation for a georeferenced raster given
    the coordinates of its upper left corner `west`, `north` and pixel
    sizes `xsize`, `ysize`.

    """
    return Affine.translation(west, north) * Affine.scale(xsize, -ysize)