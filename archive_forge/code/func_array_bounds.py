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
def array_bounds(height, width, transform):
    """Return the bounds of an array given height, width, and a transform.

    Return the `west, south, east, north` bounds of an array given
    its height, width, and an affine transform.

    """
    a, b, c, d, e, f, _, _, _ = transform
    if b == d == 0:
        west, south, east, north = (c, f + e * height, c + a * width, f)
    else:
        c0x, c0y = (c, f)
        c1x, c1y = transform * (0, height)
        c2x, c2y = transform * (width, height)
        c3x, c3y = transform * (width, 0)
        xs = (c0x, c1x, c2x, c3x)
        ys = (c0y, c1y, c2y, c3y)
        west, south, east, north = (min(xs), min(ys), max(xs), max(ys))
    return (west, south, east, north)