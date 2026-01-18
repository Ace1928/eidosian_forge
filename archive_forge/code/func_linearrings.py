import numpy as np
from shapely import Geometry, GeometryType, lib
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import multithreading_enabled
from shapely.io import from_wkt
@multithreading_enabled
def linearrings(coords, y=None, z=None, indices=None, out=None, **kwargs):
    """Create an array of linearrings.

    If the provided coords do not constitute a closed linestring, or if there
    are only 3 provided coords, the first
    coordinate is duplicated at the end to close the ring. This function will
    raise an exception if a linearring contains less than three points or if
    the terminal coordinates contain NaN (not-a-number).

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

    See also
    --------
    linestrings

    Examples
    --------
    >>> linearrings([[0, 0], [0, 1], [1, 1], [0, 0]])
    <LINEARRING (0 0, 0 1, 1 1, 0 0)>
    >>> linearrings([[0, 0], [0, 1], [1, 1]])
    <LINEARRING (0 0, 0 1, 1 1, 0 0)>

    Notes
    -----
    - Usage of the ``y`` and ``z`` arguments will prevents lazy evaluation in ``dask``.
      Instead provide the coordinates as a ``(..., 2)`` or ``(..., 3)`` array using only ``coords``.
    """
    coords = _xyz_to_coords(coords, y, z)
    if indices is None:
        return lib.linearrings(coords, out=out, **kwargs)
    else:
        return simple_geometries_1d(coords, indices, GeometryType.LINEARRING, out=out)