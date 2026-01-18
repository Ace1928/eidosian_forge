import param
import numpy as np
from holoviews import Polygons, Path
from holoviews.streams import RangeXY
from holoviews import Operation
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from ..util import polygons_to_geom_dicts, path_to_geom_dicts, shapely_v2
def bounds_to_poly(bounds):
    """
    Constructs a shapely Polygon from the provided bounds tuple.

    Parameters
    ----------
    bounds: tuple
        Tuple representing the (left, bottom, right, top) coordinates

    Returns
    -------
    polygon: shapely.geometry.Polygon
        Shapely Polygon geometry of the bounds
    """
    x0, y0, x1, y1 = bounds
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])