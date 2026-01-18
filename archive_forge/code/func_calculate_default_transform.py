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
@ensure_env
def calculate_default_transform(src_crs, dst_crs, width, height, left=None, bottom=None, right=None, top=None, gcps=None, rpcs=None, resolution=None, dst_width=None, dst_height=None, **kwargs):
    """Output dimensions and transform for a reprojection.

    Source and destination coordinate reference systems and output
    width and height are the first four, required, parameters. Source
    georeferencing can be specified using either ground control points
    (gcps) or spatial bounds (left, bottom, right, top). These two
    forms of georeferencing are mutually exclusive.

    The destination transform is anchored at the left, top coordinate.

    Destination width and height (and resolution if not provided), are
    calculated using GDAL's method for suggest warp output.

    Parameters
    ----------
    src_crs: CRS or dict
        Source coordinate reference system, in rasterio dict format.
        Example: CRS({'init': 'EPSG:4326'})
    dst_crs: CRS or dict
        Target coordinate reference system.
    width, height: int
        Source raster width and height.
    left, bottom, right, top: float, optional
        Bounding coordinates in src_crs, from the bounds property of a
        raster. Required unless using gcps.
    gcps: sequence of GroundControlPoint, optional
        Instead of a bounding box for the source, a sequence of ground
        control points may be provided.
    rpcs: RPC or dict, optional
        Instead of a bounding box for the source, rational polynomial
        coefficients may be provided.
    resolution: tuple (x resolution, y resolution) or float, optional
        Target resolution, in units of target coordinate reference
        system.
    dst_width, dst_height: int, optional
        Output file size in pixels and lines. Cannot be used together
        with resolution.
    kwargs:  dict, optional
        Additional arguments passed to transformation function.

    Returns
    -------
    transform: Affine
        Output affine transformation matrix
    width, height: int
        Output dimensions

    Notes
    -----
    Some behavior of this function is determined by the
    CHECK_WITH_INVERT_PROJ environment variable:

        YES
            constrain output raster to extents that can be inverted
            avoids visual artifacts and coordinate discontinuties.
        NO
            reproject coordinates beyond valid bound limits
    """
    if any((x is not None for x in (left, bottom, right, top))) and gcps:
        raise ValueError('Bounding values and ground control points may not be used together.')
    if any((x is not None for x in (left, bottom, right, top))) and rpcs:
        raise ValueError('Bounding values and rational polynomial coefficients may not be used together.')
    if any((x is None for x in (left, bottom, right, top))) and (not (gcps or rpcs)):
        raise ValueError('Either four bounding values, ground control points, or rational polynomial coefficients must be specified')
    if gcps and rpcs:
        raise ValueError('ground control points and rational polynomial', ' coefficients may not be used together.')
    if (dst_width is None) != (dst_height is None):
        raise ValueError('Either dst_width and dst_height must be specified or none of them.')
    if all((x is not None for x in (dst_width, dst_height))):
        dimensions = (dst_width, dst_height)
    else:
        dimensions = None
    if resolution and dimensions:
        raise ValueError('Resolution cannot be used with dst_width and dst_height.')
    dst_affine, dst_width, dst_height = _calculate_default_transform(src_crs, dst_crs, width, height, left, bottom, right, top, gcps, rpcs, **kwargs)
    if resolution:
        try:
            res = (float(resolution), float(resolution))
        except TypeError:
            res = (resolution[0], resolution[0]) if len(resolution) == 1 else resolution[0:2]
        xres = res[0]
        yres = -res[1]
        xratio = dst_affine.a / xres
        yratio = dst_affine.e / yres
        dst_affine = Affine(xres, dst_affine.b, dst_affine.c, dst_affine.d, yres, dst_affine.f)
        dst_width = ceil(dst_width * xratio)
        dst_height = ceil(dst_height * yratio)
    if dimensions:
        xratio = dst_width / dimensions[0]
        yratio = dst_height / dimensions[1]
        dst_width = dimensions[0]
        dst_height = dimensions[1]
        dst_affine = Affine(dst_affine.a * xratio, dst_affine.b, dst_affine.c, dst_affine.d, dst_affine.e * yratio, dst_affine.f)
    return (dst_affine, dst_width, dst_height)