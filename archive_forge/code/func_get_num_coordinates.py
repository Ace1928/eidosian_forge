import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_num_coordinates(geometry, **kwargs):
    """Returns the total number of coordinates in a geometry.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, Point
    >>> point = Point(0, 0)
    >>> get_num_coordinates(point)
    1
    >>> get_num_coordinates(Point(0, 0, 0))
    1
    >>> line = LineString([(0, 0), (1, 1)])
    >>> get_num_coordinates(line)
    2
    >>> get_num_coordinates(GeometryCollection([point, line]))
    3
    >>> get_num_coordinates(None)
    0
    """
    return lib.get_num_coordinates(geometry, **kwargs)