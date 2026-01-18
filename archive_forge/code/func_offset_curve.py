import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry, JOIN_STYLE
from shapely.geometry.point import Point
def offset_curve(self, distance, quad_segs=16, join_style=JOIN_STYLE.round, mitre_limit=5.0):
    """Returns a LineString or MultiLineString geometry at a distance from
        the object on its right or its left side.

        The side is determined by the sign of the `distance` parameter
        (negative for right side offset, positive for left side offset). The
        resolution of the buffer around each vertex of the object increases
        by increasing the `quad_segs` keyword parameter.

        The join style is for outside corners between line segments. Accepted
        values are JOIN_STYLE.round (1), JOIN_STYLE.mitre (2), and
        JOIN_STYLE.bevel (3).

        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled.

        Note: the behaviour regarding orientation of the resulting line
        depends on the GEOS version. With GEOS < 3.11, the line retains the
        same direction for a left offset (positive distance) or has reverse
        direction for a right offset (negative distance), and this behaviour
        was documented as such in previous Shapely versions. Starting with
        GEOS 3.11, the function tries to preserve the orientation of the
        original line.
        """
    if mitre_limit == 0.0:
        raise ValueError('Cannot compute offset from zero-length line segment')
    elif not np.isfinite(distance):
        raise ValueError('offset_curve distance must be finite')
    return shapely.offset_curve(self, distance, quad_segs, join_style, mitre_limit)