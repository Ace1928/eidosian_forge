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
def geom_to_arr(geom):
    """
    LineString, LinearRing and Polygon (exterior only?)
    """
    try:
        xy = getattr(geom, 'xy', None)
    except NotImplementedError:
        xy = None
    if xy is not None:
        return np.column_stack(xy)
    if shapely_version < Version('1.8.0'):
        if hasattr(geom, 'array_interface'):
            data = geom.array_interface()
            return np.array(data['data']).reshape(data['shape'])[:, :2]
        arr = geom.array_interface_base['data']
    else:
        arr = np.asarray(geom.exterior.coords)
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    return np.array(arr).reshape(-1, 2)