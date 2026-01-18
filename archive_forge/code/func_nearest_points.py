from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
def nearest_points(g1, g2):
    """Returns the calculated nearest points in the input geometries

    The points are returned in the same order as the input geometries.
    """
    seq = shapely.shortest_line(g1, g2)
    if seq is None:
        if g1.is_empty:
            raise ValueError('The first input geometry is empty')
        else:
            raise ValueError('The second input geometry is empty')
    p1 = shapely.get_point(seq, 0)
    p2 = shapely.get_point(seq, 1)
    return (p1, p2)