import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape=None):
    """Generate Bezier curve coordinates.

    Parameters
    ----------
    r0, c0 : int
        Coordinates of the first control point.
    r1, c1 : int
        Coordinates of the middle control point.
    r2, c2 : int
        Coordinates of the last control point.
    weight : double
        Middle control point weight, it describes the line tension.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for curves that exceed the image
        size. If None, the full extent of the curve is used.

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the Bezier curve.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The algorithm is the rational quadratic algorithm presented in
    reference [1]_.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import bezier_curve
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = bezier_curve(1, 5, 5, -2, 8, 8, 2)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return _bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape)