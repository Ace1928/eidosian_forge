import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def line_aa(r0, c0, r1, c1):
    """Generate anti-aliased line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    References
    ----------
    .. [1] A Rasterizing Algorithm for Drawing Curves, A. Zingl, 2012
           http://members.chello.at/easyfilter/Bresenham.pdf

    Examples
    --------
    >>> from skimage.draw import line_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = line_aa(1, 1, 8, 8)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0, 255,  74,   0,   0,   0,   0,   0,   0,   0],
           [  0,  74, 255,  74,   0,   0,   0,   0,   0,   0],
           [  0,   0,  74, 255,  74,   0,   0,   0,   0,   0],
           [  0,   0,   0,  74, 255,  74,   0,   0,   0,   0],
           [  0,   0,   0,   0,  74, 255,  74,   0,   0,   0],
           [  0,   0,   0,   0,   0,  74, 255,  74,   0,   0],
           [  0,   0,   0,   0,   0,   0,  74, 255,  74,   0],
           [  0,   0,   0,   0,   0,   0,   0,  74, 255,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    return _line_aa(r0, c0, r1, c1)