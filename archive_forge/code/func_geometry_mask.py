import logging
import math
import os
import warnings
import numpy as np
import rasterio
from rasterio import warp
from rasterio._base import DatasetBase
from rasterio._features import _shapes, _sieve, _rasterize, _bounds
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype, _getnpdtype
from rasterio.enums import MergeAlg
from rasterio.env import ensure_env, GDALVersion
from rasterio.errors import ShapeSkipWarning
from rasterio.rio.helpers import coords
from rasterio.transform import Affine
from rasterio.transform import IDENTITY, guard_transform
from rasterio.windows import Window
@ensure_env
def geometry_mask(geometries, out_shape, transform, all_touched=False, invert=False):
    """Create a mask from shapes.

    By default, mask is intended for use as a
    numpy mask, where pixels that overlap shapes are False.

    Parameters
    ----------
    geometries : iterable over geometries (GeoJSON-like objects)
    out_shape : tuple or list
        Shape of output numpy ndarray.
    transform : Affine transformation object
        Transformation from pixel coordinates of `source` to the
        coordinate system of the input `shapes`. See the `transform`
        property of dataset objects.
    all_touched : boolean, optional
        If True, all pixels touched by geometries will be burned in.  If
        false, only pixels whose center is within the polygon or that
        are selected by Bresenham's line algorithm will be burned in.
    invert: boolean, optional
        If True, mask will be True for pixels that overlap shapes.
        False by default.

    Returns
    -------
    numpy ndarray of type 'bool'
        Result

    Notes
    -----
    See rasterize() for performance notes.

    """
    fill, mask_value = (0, 1) if invert else (1, 0)
    return rasterize(geometries, out_shape=out_shape, transform=transform, all_touched=all_touched, fill=fill, default_value=mask_value).astype('bool')