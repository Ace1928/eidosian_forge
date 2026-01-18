import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.7.0')
@multithreading_enabled
def get_z(point, **kwargs):
    """Returns the z-coordinate of a point.

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries or geometries without 3rd dimension will result
        in NaN being returned.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_x, get_y

    Examples
    --------
    >>> from shapely import MultiPoint, Point
    >>> get_z(Point(1, 2, 3))
    3.0
    >>> get_z(Point(1, 2))
    nan
    >>> get_z(MultiPoint([(1, 1, 1), (2, 2, 2)]))
    nan
    """
    return lib.get_z(point, **kwargs)