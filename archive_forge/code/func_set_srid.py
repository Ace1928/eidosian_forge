import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def set_srid(geometry, srid, **kwargs):
    """Returns a geometry with its SRID set.

    Parameters
    ----------
    geometry : Geometry or array_like
    srid : int
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_srid

    Examples
    --------
    >>> from shapely import Point
    >>> point = Point(0, 0)
    >>> get_srid(point)
    0
    >>> with_srid = set_srid(point, 4326)
    >>> get_srid(with_srid)
    4326
    """
    return lib.set_srid(geometry, np.intc(srid), **kwargs)