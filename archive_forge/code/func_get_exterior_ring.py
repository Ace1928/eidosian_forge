import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_exterior_ring(geometry, **kwargs):
    """Returns the exterior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_interior_ring

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> get_exterior_ring(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    <LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>
    >>> get_exterior_ring(Point(1, 1)) is None
    True
    """
    return lib.get_exterior_ring(geometry, **kwargs)