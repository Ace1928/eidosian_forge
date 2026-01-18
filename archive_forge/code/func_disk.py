import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def disk(center, radius, *, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    center : tuple
        Center coordinate of disk.
    radius : double
        Radius of disk.
    shape : tuple, optional
        Image shape as a tuple of size 2. Determines the maximum
        extent of output pixel coordinates. This is useful for disks that
        exceed the image size. If None, the full extent of the disk is used.
        The  shape might result in negative coordinates and wraparound
        behaviour.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of disk.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.draw import disk
    >>> shape = (4, 4)
    >>> img = np.zeros(shape, dtype=np.uint8)
    >>> rr, cc = disk((0, 0), 2, shape=shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=uint8)
    >>> img = np.zeros(shape, dtype=np.uint8)
    >>> # Negative coordinates in rr and cc perform a wraparound
    >>> rr, cc = disk((0, 0), 2, shape=None)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 0, 1],
           [1, 1, 0, 1],
           [0, 0, 0, 0],
           [1, 1, 0, 1]], dtype=uint8)
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = disk((4, 4), 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    r, c = center
    return ellipse(r, c, radius, radius, shape)