import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def force_3d(geometry, z=0.0, **kwargs):
    """Forces the dimensionality of a geometry to 3D.

    2D geometries will get the provided Z coordinate; Z coordinates of 3D geometries
    are unchanged (unless they are nan).

    Note that for empty geometries, 3D is only supported since GEOS 3.9 and then
    still only for simple geometries (non-collections).

    Parameters
    ----------
    geometry : Geometry or array_like
    z : float or array_like, default 0.0
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> force_3d(Point(0, 0), z=3)
    <POINT Z (0 0 3)>
    >>> force_3d(Point(0, 0, 0), z=3)
    <POINT Z (0 0 0)>
    >>> force_3d(LineString([(0, 0), (0, 1), (1, 1)]))
    <LINESTRING Z (0 0 0, 0 1 0, 1 1 0)>
    >>> force_3d(None) is None
    True
    """
    if np.isnan(z).any():
        raise ValueError('It is not allowed to set the Z coordinate to NaN.')
    return lib.force_3d(geometry, z, **kwargs)