import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def is_geometry(geometry, **kwargs):
    """Returns True if the object is a geometry

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    is_missing : check if an object is missing (None)
    is_valid_input : check if an object is a geometry or None

    Examples
    --------
    >>> from shapely import GeometryCollection, Point
    >>> is_geometry(Point(0, 0))
    True
    >>> is_geometry(GeometryCollection())
    True
    >>> is_geometry(None)
    False
    >>> is_geometry("text")
    False
    """
    return lib.is_geometry(geometry, **kwargs)