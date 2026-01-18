from shapely import lib
from shapely.decorators import multithreading_enabled
from shapely.errors import UnsupportedGEOSVersionError
@multithreading_enabled
def shortest_line(a, b, **kwargs):
    """
    Returns the shortest line between two geometries.

    The resulting line consists of two points, representing the nearest
    points between the geometry pair. The line always starts in the first
    geometry `a` and ends in he second geometry `b`. The endpoints of the
    line will not necessarily be existing vertices of the input geometries
    `a` and `b`, but can also be a point along a line segment.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument) (for GEOS>=3.9)

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> line2 = LineString([(0, 3), (3, 0), (5, 3)])
    >>> shortest_line(line1, line2)
    <LINESTRING (1 1, 1.5 1.5)>
    """
    return lib.shortest_line(a, b, **kwargs)