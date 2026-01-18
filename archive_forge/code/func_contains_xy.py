import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def contains_xy(geom, x, y=None, **kwargs):
    """
    Returns True if the Point (x, y) is completely inside geometry A.

    This is a special-case (and faster) variant of the `contains` function
    which avoids having to create a Point object if you start from x/y
    coordinates.

    Note that in the case of points, the `contains_properly` predicate is
    equivalent to `contains`.

    See the docstring of `contains` for more details about the predicate.

    Parameters
    ----------
    geom : Geometry or array_like
    x, y : float or array_like
        Coordinates as separate x and y arrays, or a single array of
        coordinate x, y tuples.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    contains : variant taking two geometries as input

    Notes
    -----
    If you compare a small number of polygons or lines with many points,
    it can be beneficial to prepare the geometries in advance using
    :func:`shapely.prepare`.

    Examples
    --------
    >>> from shapely import Point, Polygon
    >>> area = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> contains(area, Point(0.5, 0.5))
    True
    >>> contains_xy(area, 0.5, 0.5)
    True
    """
    if y is None:
        coords = np.asarray(x)
        x, y = (coords[:, 0], coords[:, 1])
    return lib.contains_xy(geom, x, y, **kwargs)