import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def clip_by_rect(geometry, xmin, ymin, xmax, ymax, **kwargs):
    """
    Returns the portion of a geometry within a rectangle.

    The geometry is clipped in a fast but possibly dirty way. The output is
    not guaranteed to be valid. No exceptions will be raised for topological
    errors.

    Note: empty geometries or geometries that do not overlap with the
    specified bounds will result in GEOMETRYCOLLECTION EMPTY.

    Parameters
    ----------
    geometry : Geometry or array_like
        The geometry to be clipped
    xmin : float
        Minimum x value of the rectangle
    ymin : float
        Minimum y value of the rectangle
    xmax : float
        Maximum x value of the rectangle
    ymax : float
        Maximum y value of the rectangle
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (10, 10)])
    >>> clip_by_rect(line, 0., 0., 1., 1.)
    <LINESTRING (0 0, 1 1)>
    >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> clip_by_rect(polygon, 0., 0., 1., 1.)
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    if not all((np.isscalar(val) for val in [xmin, ymin, xmax, ymax])):
        raise TypeError('xmin/ymin/xmax/ymax only accepts scalar values')
    return lib.clip_by_rect(geometry, np.double(xmin), np.double(ymin), np.double(xmax), np.double(ymax), **kwargs)