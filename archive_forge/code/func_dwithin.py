import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def dwithin(self, other, distance):
    """
        Returns True if geometry is within a given distance from the other, else False.

        Refer to `shapely.dwithin` for full documentation.
        """
    return _maybe_unpack(shapely.dwithin(self, other, distance))