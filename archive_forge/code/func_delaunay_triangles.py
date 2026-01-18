import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def delaunay_triangles(geometry, tolerance=0.0, only_edges=False, **kwargs):
    """Computes a Delaunay triangulation around the vertices of an input
    geometry.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see only_edges). Returns an None if an input geometry
    contains less than 3 vertices.

    Parameters
    ----------
    geometry : Geometry or array_like
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
    only_edges : bool or array_like, default False
        If set to True, the triangulation will return a collection of
        linestrings instead of polygons.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Polygon
    >>> points = MultiPoint([(50, 30), (60, 30), (100, 100)])
    >>> delaunay_triangles(points)
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(points, only_edges=True)
    <MULTILINESTRING ((50 30, 100 100), (50 30, 60 30), ...>
    >>> delaunay_triangles(MultiPoint([(50, 30), (51, 30), (60, 30), (100, 100)]), tolerance=2)
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(Polygon([(50, 30), (60, 30), (100, 100), (50, 30)]))
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(LineString([(50, 30), (60, 30), (100, 100)]))
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(GeometryCollection([]))
    <GEOMETRYCOLLECTION EMPTY>
    """
    return lib.delaunay_triangles(geometry, tolerance, only_edges, **kwargs)