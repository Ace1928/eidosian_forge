import numpy as np
import shapely
from shapely.algorithms.cga import is_ccw_impl, signed_area
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
@classmethod
def from_bounds(cls, xmin, ymin, xmax, ymax):
    """Construct a `Polygon()` from spatial bounds."""
    return cls([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])