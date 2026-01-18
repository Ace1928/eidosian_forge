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
def is_multi_geometry(geom):
    """
    Whether the shapely geometry is a Multi or Collection type.
    """
    return 'Multi' in geom.geom_type or 'Collection' in geom.geom_type