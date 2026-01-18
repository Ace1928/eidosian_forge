import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_interior_ring(geometry, index, **kwargs):
    """Returns the nth interior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the interior rings backwards.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_exterior_ring
    get_num_interior_rings

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> get_interior_ring(polygon_with_hole, 0)
    <LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> get_interior_ring(polygon_with_hole, 1) is None
    True
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> get_interior_ring(polygon, 0) is None
    True
    >>> get_interior_ring(Point(0, 0), 0) is None
    True
    """
    return lib.get_interior_ring(geometry, np.intc(index), **kwargs)