import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
def circle_perimeter_aa(r, c, radius, shape=None):
    """Generate anti-aliased circle perimeter coordinates.

    Parameters
    ----------
    r, c : int
        Centre coordinate of circle.
    radius : int
        Radius of circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for circles that exceed the image
        size. If None, the full extent of the circle is used.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.

    Returns
    -------
    rr, cc, val : (N,) ndarray (int, int, float)
        Indices of pixels (`rr`, `cc`) and intensity values (`val`).
        ``img[rr, cc] = val``.

    Notes
    -----
    Wu's method draws anti-aliased circle. This implementation doesn't use
    lookup table optimization.

    Use the function ``draw.set_color`` to apply ``circle_perimeter_aa``
    results to color images.

    References
    ----------
    .. [1] X. Wu, "An efficient antialiasing technique", In ACM SIGGRAPH
           Computer Graphics, 25 (1991) 143-152.

    Examples
    --------
    >>> from skimage.draw import circle_perimeter_aa
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc, val = circle_perimeter_aa(4, 4, 3)
    >>> img[rr, cc] = val * 255
    >>> img
    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0, 255,   0,   0,   0,   0,   0, 255,   0,   0],
           [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
           [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
           [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)

    >>> from skimage import data, draw
    >>> image = data.chelsea()
    >>> rr, cc, val = draw.circle_perimeter_aa(r=100, c=100, radius=75)
    >>> draw.set_color(image, (rr, cc), [1, 0, 0], alpha=val)
    """
    return _circle_perimeter_aa(r, c, radius, shape)