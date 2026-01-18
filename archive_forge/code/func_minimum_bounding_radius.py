import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.8.0')
@multithreading_enabled
def minimum_bounding_radius(geometry, **kwargs):
    """Computes the radius of the minimum bounding circle that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.


    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> minimum_bounding_radius(Polygon([(0, 5), (5, 10), (10, 5), (5, 0), (0, 5)]))
    5.0
    >>> minimum_bounding_radius(LineString([(1, 1), (1, 10)]))
    4.5
    >>> minimum_bounding_radius(MultiPoint([(2, 2), (4, 2)]))
    1.0
    >>> minimum_bounding_radius(Point(0, 1))
    0.0
    >>> minimum_bounding_radius(GeometryCollection())
    0.0

    See also
    --------
    minimum_bounding_circle
    """
    return lib.minimum_bounding_radius(geometry, **kwargs)