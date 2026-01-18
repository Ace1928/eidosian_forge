from typing import Any, Iterable, Union
import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.decorators import requires_geos, UnsupportedGEOSVersionError
from shapely.geometry.base import BaseGeometry
from shapely.predicates import is_empty, is_missing
@requires_geos('3.6.0')
def query_nearest(self, geometry, max_distance=None, return_distance=False, exclusive=False, all_matches=True):
    """Return the index of the nearest geometries in the tree for each input
        geometry based on distance within two-dimensional Cartesian space.

        This distance will be 0 when input geometries intersect tree geometries.

        If there are multiple equidistant or intersected geometries in tree and
        `all_matches` is True (the default), all matching tree geometries are
        returned; otherwise only the first matching tree geometry is returned.
        Tree indices are returned in the order they are visited for each input
        geometry and may not be in ascending index order; no meaningful order is
        implied.

        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.  Only those
        input geometries with at least one tree geometry within +/- max_distance
        beyond their envelope will be evaluated.  However, using a large
        max_distance may have a negative performance impact because many tree
        geometries will be queried for each input geometry.

        The distance, if returned, will be 0 for any intersected geometries in
        the tree.

        Any geometry that is None or empty in the input geometries is omitted
        from the output.  Any Z values present in input geometries are ignored
        when finding nearest tree geometries.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0.
        return_distance : bool, default False
            If True, will return distances in addition to indices.
        exclusive : bool, default False
            If True, the nearest tree geometries that are equal to the input
            geometry will not be returned.
        all_matches : bool, default True
            If True, all equidistant and intersected geometries will be returned
            for each input geometry.
            If False, only the first nearest geometry will be returned.

        Returns
        -------
        tree indices or tuple of (tree indices, distances) if geometry is a scalar
            indices is an ndarray of shape (n, ) and distances (if present) an
            ndarray of shape (n, )

        OR

        indices or tuple of (indices, distances)
            indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        See also
        --------
        nearest: returns singular nearest geometry for each input

        Examples
        --------
        >>> import numpy as np
        >>> from shapely import box, Point
        >>> points = [Point(0, 0), Point(1, 1), Point(2,2), Point(3, 3)]
        >>> tree = STRtree(points)

        Find the nearest tree geometries to a scalar geometry:

        >>> indices = tree.query_nearest(Point(0.25, 0.25))
        >>> indices.tolist()
        [0]

        Retrieve the tree geometries by results of query:

        >>> tree.geometries.take(indices).tolist()
        [<POINT (0 0)>]

        Find the nearest tree geometries to an array of geometries:

        >>> query_points = np.array([Point(2.25, 2.25), Point(1, 1)])
        >>> arr_indices = tree.query_nearest(query_points)
        >>> arr_indices.tolist()
        [[0, 1], [2, 1]]

        Or transpose to get all pairs of input and tree indices:

        >>> arr_indices.T.tolist()
        [[0, 2], [1, 1]]

        Retrieve all pairs of input and tree geometries:

        >>> list(zip(query_points.take(arr_indices[0]), tree.geometries.take(arr_indices[1])))
        [(<POINT (2.25 2.25)>, <POINT (2 2)>), (<POINT (1 1)>, <POINT (1 1)>)]

        All intersecting geometries in the tree are returned by default:

        >>> tree.query_nearest(box(1,1,3,3)).tolist()
        [1, 2, 3]

        Set all_matches to False to to return a single match per input geometry:

        >>> tree.query_nearest(box(1,1,3,3), all_matches=False).tolist()
        [1]

        Return the distance to each nearest tree geometry:

        >>> index, distance = tree.query_nearest(Point(0.5, 0.5), return_distance=True)
        >>> index.tolist()
        [0, 1]
        >>> distance.round(4).tolist()
        [0.7071, 0.7071]

        Return the distance for each input and nearest tree geometry for an array
        of geometries:

        >>> indices, distance = tree.query_nearest([Point(0.5, 0.5), Point(1, 1)], return_distance=True)
        >>> indices.tolist()
        [[0, 0, 1], [0, 1, 1]]
        >>> distance.round(4).tolist()
        [0.7071, 0.7071, 0.0]

        Retrieve custom items associated with tree geometries (records can
        be in whatever data structure so long as geometries and custom data
        can be extracted into arrays of the same length and order):

        >>> records = [
        ...     {"geometry": Point(0, 0), "value": "A"},
        ...     {"geometry": Point(2, 2), "value": "B"}
        ... ]
        >>> tree = STRtree([record["geometry"] for record in records])
        >>> items = np.array([record["value"] for record in records])
        >>> items.take(tree.query_nearest(Point(0.5, 0.5))).tolist()
        ['A']
        """
    geometry = np.asarray(geometry, dtype=object)
    is_scalar = False
    if geometry.ndim == 0:
        geometry = np.expand_dims(geometry, 0)
        is_scalar = True
    if max_distance is not None:
        if not np.isscalar(max_distance):
            raise ValueError('max_distance parameter only accepts scalar values')
        if max_distance <= 0:
            raise ValueError('max_distance must be greater than 0')
    max_distance = max_distance or 0
    if not np.isscalar(exclusive):
        raise ValueError('exclusive parameter only accepts scalar values')
    if exclusive not in {True, False}:
        raise ValueError('exclusive parameter must be boolean')
    if not np.isscalar(all_matches):
        raise ValueError('all_matches parameter only accepts scalar values')
    if all_matches not in {True, False}:
        raise ValueError('all_matches parameter must be boolean')
    results = self._tree.query_nearest(geometry, max_distance, exclusive, all_matches)
    if is_scalar:
        if not return_distance:
            return results[0][1]
        else:
            return (results[0][1], results[1])
    if not return_distance:
        return results[0]
    return results