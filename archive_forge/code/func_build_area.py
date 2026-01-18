import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.8.0')
@multithreading_enabled
def build_area(geometry, **kwargs):
    """Creates an areal geometry formed by the constituent linework of given geometry.

    Equivalent of the PostGIS ST_BuildArea() function.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import GeometryCollection, Polygon
    >>> polygon1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    >>> polygon2 = Polygon([(1, 1), (1, 2), (2, 2), (1, 1)])
    >>> build_area(GeometryCollection([polygon1, polygon2]))
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 2 2, 1 2, 1 1))>
    """
    return lib.build_area(geometry, **kwargs)