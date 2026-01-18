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
def guard_transform(transform):
    """Return an Affine transformation instance."""
    if not isinstance(transform, Affine):
        if tastes_like_gdal(transform):
            raise TypeError('GDAL-style transforms have been deprecated.  This exception will be raised for a period of time to highlight potentially confusing errors, but will eventually be removed.')
        else:
            transform = Affine(*transform)
    return transform