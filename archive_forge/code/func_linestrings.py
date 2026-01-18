import numpy as np
from shapely import Geometry, GeometryType, lib
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import multithreading_enabled
from shapely.io import from_wkt
@multithreading_enabled
def linestrings(coords, y=None, z=None, indices=None, out=None, **kwargs):
    """Create an array of linestrings.

    This function will raise an exception if a linestring contains less than
    two points.

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if ``y``
        is provided, an array of lists of x coordinates
    y : array_like, optional
    z : array_like, optional
    indices : array_like, optional
        Indices into the target array where input coordinates belong. If
        provided, the coords should be 2D with shape (N, 2) or (N, 3) and
        indices should be an array of shape (N,) with integers in increasing
        order. Missing indices result in a ValueError unless ``out`` is
        provided, in which case the original value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    Examples
    --------
    >>> linestrings([[[0, 1], [4, 5]], [[2, 3], [5, 6]]]).tolist()
    [<LINESTRING (0 1, 4 5)>, <LINESTRING (2 3, 5 6)>]
    >>> linestrings([[0, 1], [4, 5], [2, 3], [5, 6], [7, 8]], indices=[0, 0, 1, 1, 1]).tolist()
    [<LINESTRING (0 1, 4 5)>, <LINESTRING (2 3, 5 6, 7 8)>]

    Notes
    -----
    - Usage of the ``y`` and ``z`` arguments will prevents lazy evaluation in ``dask``.
      Instead provide the coordinates as a ``(..., 2)`` or ``(..., 3)`` array using only ``coords``.
    """
    coords = _xyz_to_coords(coords, y, z)
    if indices is None:
        return lib.linestrings(coords, out=out, **kwargs)
    else:
        return simple_geometries_1d(coords, indices, GeometryType.LINESTRING, out=out)