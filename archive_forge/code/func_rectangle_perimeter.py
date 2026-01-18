import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
@require('matplotlib', '>=3.3')
def rectangle_perimeter(start, end=None, extent=None, shape=None, clip=False):
    """Generate coordinates of pixels that are exactly around a rectangle.

    Parameters
    ----------
    start : tuple
        Origin point of the inner rectangle, e.g., ``(row, column)``.
    end : tuple
        End point of the inner rectangle ``(row, column)``.
        For a 2D matrix, the slice defined by inner the rectangle is
        ``[start:(end+1)]``.
        Either `end` or `extent` must be specified.
    extent : tuple
        The extent (size) of the inner rectangle.  E.g.,
        ``(num_rows, num_cols)``.
        Either `end` or `extent` must be specified.
        Negative extents are permitted. See `rectangle` to better
        understand how they behave.
    shape : tuple, optional
        Image shape used to determine the maximum bounds of the output
        coordinates. This is useful for clipping perimeters that exceed
        the image size. By default, no clipping is done.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.
    clip : bool, optional
        Whether to clip the perimeter to the provided shape. If this is set
        to True, the drawn figure will always be a closed polygon with all
        edges visible.

    Returns
    -------
    coords : array of int, shape (2, Npoints)
        The coordinates of all pixels in the rectangle.

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> import numpy as np
    >>> from skimage.draw import rectangle_perimeter
    >>> img = np.zeros((5, 6), dtype=np.uint8)
    >>> start = (2, 3)
    >>> end = (3, 4)
    >>> rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 1, 1, 1]], dtype=uint8)

    >>> img = np.zeros((5, 5), dtype=np.uint8)
    >>> r, c = rectangle_perimeter(start, (10, 10), shape=img.shape, clip=True)
    >>> img[r, c] = 1
    >>> img
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 1]], dtype=uint8)

    """
    top_left, bottom_right = _rectangle_slice(start=start, end=end, extent=extent)
    top_left -= 1
    r = [top_left[0], top_left[0], bottom_right[0], bottom_right[0], top_left[0]]
    c = [top_left[1], bottom_right[1], bottom_right[1], top_left[1], top_left[1]]
    return polygon_perimeter(r, c, shape=shape, clip=clip)