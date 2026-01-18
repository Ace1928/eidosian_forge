import numpy as np
from shapely import GeometryType, lib
from shapely.decorators import multithreading_enabled, requires_geos
from shapely.errors import UnsupportedGEOSVersionError
@multithreading_enabled
def symmetric_difference_all(geometries, axis=None, **kwargs):
    """Returns the symmetric difference of multiple geometries.

    This function ignores None values when other Geometry elements are present.
    If all elements of the given axis are None an empty GeometryCollection is
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
    symmetric_difference

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (2, 2)])
    >>> line2 = LineString([(1, 1), (3, 3)])
    >>> symmetric_difference_all([line1, line2])
    <MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference_all([[line1, line2, None]], axis=1).tolist()
    [<MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>]
    >>> symmetric_difference_all([line1, None])
    <LINESTRING (0 0, 2 2)>
    >>> symmetric_difference_all([None, None])
    <GEOMETRYCOLLECTION EMPTY>
    """
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(geometries, axis=axis, start=geometries.ndim)
    return lib.symmetric_difference_all(geometries, **kwargs)