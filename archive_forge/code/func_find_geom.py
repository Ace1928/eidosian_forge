import param
import numpy as np
from holoviews import Polygons, Path
from holoviews.streams import RangeXY
from holoviews import Operation
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from ..util import polygons_to_geom_dicts, path_to_geom_dicts, shapely_v2
def find_geom(geom, geoms):
    """
    Returns the index of a geometry in a list of geometries avoiding
    expensive equality checks of `in` operator.
    """
    for i, g in enumerate(geoms):
        if g is geom:
            return i