import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def covered_by(self, other):
    """Returns True if the geometry is covered by the other, else False"""
    return _maybe_unpack(shapely.covered_by(self, other))