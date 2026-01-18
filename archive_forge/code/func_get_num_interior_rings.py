import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def get_num_interior_rings(geometry, **kwargs):
    """Returns number of internal rings in a polygon

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of interior rings in non-polygons equals zero.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_exterior_ring
    get_interior_ring

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> get_num_interior_rings(polygon)
    0
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> get_num_interior_rings(polygon_with_hole)
    1
    >>> get_num_interior_rings(Point(0, 0))
    0
    >>> get_num_interior_rings(None)
    0
    """
    return lib.get_num_interior_rings(geometry, **kwargs)