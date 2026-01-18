import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
def polygonize(geometries, **kwargs):
    """Creates polygons formed from the linework of a set of Geometries.

    Polygonizes an array of Geometries that contain linework which
    represents the edges of a planar graph. Any type of Geometry may be
    provided as input; only the constituent lines and rings will be used to
    create the output polygons.

    Lines or rings that when combined do not completely close a polygon
    will result in an empty GeometryCollection.  Duplicate segments are
    ignored.

    This function returns the polygons within a GeometryCollection.
    Individual Polygons can be obtained using ``get_geometry`` to get
    a single polygon or ``get_parts`` to get an array of polygons.
    MultiPolygons can be constructed from the output using
    ``shapely.multipolygons(shapely.get_parts(shapely.polygonize(geometries)))``.

    Parameters
    ----------
    geometries : array_like
        An array of geometries.
    axis : int
        Axis along which the geometries are polygonized.
        The default is to perform a reduction over the last dimension
        of the input array. A 1D array results in a scalar geometry.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    GeometryCollection or array of GeometryCollections

    See Also
    --------
    get_parts, get_geometry
    polygonize_full
    node

    Examples
    --------
    >>> from shapely import LineString
    >>> lines = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(0, 0), (0, 1)]),
    ...     LineString([(0, 1), (1, 1)])
    ... ]
    >>> polygonize(lines)
    <GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))>
    """
    return lib.polygonize(geometries, **kwargs)