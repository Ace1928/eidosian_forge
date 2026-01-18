import numpy as np
import shapely
import shapely.geometry as sgeom
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from holoviews.element import Tiles
from packaging.version import Version
from shapely.geometry import (
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import transform
from ._warnings import warn
def geo_mesh(element):
    """
    Get mesh data from a 2D Element ensuring that if the data is
    on a cylindrical coordinate system and wraps globally that data
    actually wraps around.
    """
    if len(element.vdims) > 1:
        xs, ys = (element.dimension_values(i, False, False) for i in range(2))
        zs = np.dstack([element.dimension_values(i, False, False) for i in range(2, 2 + len(element.vdims))])
    else:
        xs, ys, zs = (element.dimension_values(i, False, False) for i in range(3))
    lon0, lon1 = element.range(0)
    if isinstance(element.crs, ccrs._CylindricalProjection) and lon1 - lon0 == 360:
        xs = np.append(xs, xs[0:1] + 360, axis=0)
        zs = np.ma.concatenate([zs, zs[:, 0:1]], axis=1)
    return (xs, ys, zs)