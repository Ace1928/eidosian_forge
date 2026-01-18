import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def line_locate_point(self, other, normalized=False):
    """Returns the distance along this geometry to a point nearest the
        specified point

        If the normalized arg is True, return the distance normalized to the
        length of the linear geometry.

        Alias of `project`.
        """
    return shapely.line_locate_point(self, other, normalized=normalized)