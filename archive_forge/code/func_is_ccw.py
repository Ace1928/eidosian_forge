import numpy as np
import shapely
from shapely.algorithms.cga import is_ccw_impl, signed_area
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
@property
def is_ccw(self):
    """True is the ring is oriented counter clock-wise"""
    return bool(is_ccw_impl()(self))