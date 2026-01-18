import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_point(geometry, index, **kwargs):
    """Returns the nth point of a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the linestring backwards.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_num_points

    Examples
    --------
    >>> from shapely import LinearRing, LineString, MultiPoint, Point
    >>> line = LineString([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> get_point(line, 1)
    <POINT (1 1)>
    >>> get_point(line, -2)
    <POINT (2 2)>
    >>> get_point(line, [0, 3]).tolist()
    [<POINT (0 0)>, <POINT (3 3)>]

    The functcion works the same for LinearRing input:

    >>> get_point(LinearRing([(0, 0), (1, 1), (2, 2), (0, 0)]), 1)
    <POINT (1 1)>

    For non-linear geometries it returns None:

    >>> get_point(MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]), 1) is None
    True
    >>> get_point(Point(1, 1), 0) is None
    True
    """
    return lib.get_point(geometry, np.intc(index), **kwargs)