import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def line_interpolate_point(self, distance, normalized=False):
    """Return a point at the specified distance along a linear geometry

        Negative length values are taken as measured in the reverse
        direction from the end of the geometry. Out-of-range index
        values are handled by clamping them to the valid range of values.
        If the normalized arg is True, the distance will be interpreted as a
        fraction of the geometry's length.

        Alias of `interpolate`.
        """
    return shapely.line_interpolate_point(self, distance, normalized=normalized)