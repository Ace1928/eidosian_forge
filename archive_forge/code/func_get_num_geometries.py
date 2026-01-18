import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_num_geometries(geometry, **kwargs):
    """Returns number of geometries in a collection.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of geometries in points, linestrings, linearrings and
        polygons equals one.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_num_points
    get_geometry

    Examples
    --------
    >>> from shapely import MultiPoint, Point
    >>> get_num_geometries(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]))
    4
    >>> get_num_geometries(Point(1, 1))
    1
    >>> get_num_geometries(None)
    0
    """
    return lib.get_num_geometries(geometry, **kwargs)