import numpy as np
from shapely import GeometryType, lib
from shapely.decorators import multithreading_enabled, requires_geos
from shapely.errors import UnsupportedGEOSVersionError
@requires_geos('3.8.0')
@multithreading_enabled
def coverage_union_all(geometries, axis=None, **kwargs):
    """Returns the union of multiple polygons of a geometry collection.
    This is an optimized version of union which assumes the polygons
    to be non-overlapping.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None, an empty MultiPolygon is
    returned.

    Parameters
    ----------
    geometries : array_like
    axis : int, optional
        Axis along which the operation is performed. The default (None)
        performs the operation over all axes, returning a scalar value.
        Axis may be negative, in which case it counts from the last to the
        first axis.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    coverage_union

    Examples
    --------
    >>> from shapely import normalize, Polygon
    >>> polygon_1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    >>> polygon_2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0), (1, 0)])
    >>> normalize(coverage_union_all([polygon_1, polygon_2]))
    <POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    >>> normalize(coverage_union_all([polygon_1, None]))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> normalize(coverage_union_all([None, None]))
    <MULTIPOLYGON EMPTY>
    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(np.asarray(geometries), axis=axis, start=geometries.ndim)
    collections = lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)
    return lib.coverage_union(collections, **kwargs)