from math import ceil, floor
from affine import Affine
import numpy as np
import rasterio
from rasterio._base import _transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.env import ensure_env, require_gdal_version
from rasterio.errors import TransformError, RPCError
from rasterio.transform import array_bounds
from rasterio._warp import (
def aligned_target(transform, width, height, resolution):
    """Aligns target to specified resolution

    Parameters
    ----------
    transform : Affine
        Input affine transformation matrix
    width, height: int
        Input dimensions
    resolution: tuple (x resolution, y resolution) or float
        Target resolution, in units of target coordinate reference
        system.

    Returns
    -------
    transform: Affine
        Output affine transformation matrix
    width, height: int
        Output dimensions

    """
    if isinstance(resolution, (float, int)):
        res = (float(resolution), float(resolution))
    else:
        res = resolution
    xmin = transform.xoff
    ymin = transform.yoff + height * transform.e
    xmax = transform.xoff + width * transform.a
    ymax = transform.yoff
    xmin = floor(xmin / res[0]) * res[0]
    xmax = ceil(xmax / res[0]) * res[0]
    ymin = floor(ymin / res[1]) * res[1]
    ymax = ceil(ymax / res[1]) * res[1]
    dst_transform = Affine(res[0], 0, xmin, 0, -res[1], ymax)
    dst_width = max(int(ceil((xmax - xmin) / res[0])), 1)
    dst_height = max(int(ceil((ymax - ymin) / res[1])), 1)
    return (dst_transform, dst_width, dst_height)