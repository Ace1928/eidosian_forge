import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def force_2d(geometry, **kwargs):
    """Forces the dimensionality of a geometry to 2D.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon, from_wkt
    >>> force_2d(Point(0, 0, 1))
    <POINT (0 0)>
    >>> force_2d(Point(0, 0))
    <POINT (0 0)>
    >>> force_2d(LineString([(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
    <LINESTRING (0 0, 0 1, 1 1)>
    >>> force_2d(from_wkt("POLYGON Z EMPTY"))
    <POLYGON EMPTY>
    >>> force_2d(None) is None
    True
    """
    return lib.force_2d(geometry, **kwargs)