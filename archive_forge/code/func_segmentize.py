import re
from warnings import warn
import numpy as np
import shapely
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
def segmentize(self, max_segment_length):
    """Adds vertices to line segments based on maximum segment length.

        Additional vertices will be added to every line segment in an input geometry
        so that segments are no longer than the provided maximum segment length. New
        vertices will evenly subdivide each segment.

        Only linear components of input geometries are densified; other geometries
        are returned unmodified.

        Parameters
        ----------
        max_segment_length : float or array_like
            Additional vertices will be added so that all line segments are no
            longer this value.  Must be greater than 0.

        Examples
        --------
        >>> from shapely import LineString, Polygon
        >>> LineString([(0, 0), (0, 10)]).segmentize(max_segment_length=5)
        <LINESTRING (0 0, 0 5, 0 10)>
        >>> Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]).segmentize(max_segment_length=5)
        <POLYGON ((0 0, 5 0, 10 0, 10 5, 10 10, 5 10, 0 10, 0 5, 0 0))>
        """
    return shapely.segmentize(self, max_segment_length)