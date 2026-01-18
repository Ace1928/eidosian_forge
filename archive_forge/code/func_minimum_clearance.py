import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
@property
def minimum_clearance(self):
    """Unitless distance by which a node could be moved to produce an invalid geometry (float)"""
    return float(shapely.minimum_clearance(self))