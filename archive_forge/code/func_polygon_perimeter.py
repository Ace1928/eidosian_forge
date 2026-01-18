import numpy as np
from .._shared._geometry import polygon_clip
from .._shared.version_requirements import require
from .._shared.compat import NP_COPY_IF_NEEDED
from ._draw import (
@require('matplotlib', '>=3.3')
def polygon_perimeter(r, c, shape=None, clip=False):
    """Generate polygon perimeter coordinates.

    Parameters
    ----------
    r : (N,) ndarray
        Row coordinates of vertices of polygon.
    c : (N,) ndarray
        Column coordinates of vertices of polygon.
    shape : tuple, optional
        Image shape which is used to determine maximum extents of output pixel
        coordinates. This is useful for polygons that exceed the image size.
        If None, the full extents of the polygon is used.  Must be at least
        length 2. Only the first two values are used to determine the extent of
        the input image.
    clip : bool, optional
        Whether to clip the polygon to the provided shape.  If this is set
        to True, the drawn figure will always be a closed polygon with all
        edges visible.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> from skimage.draw import polygon_perimeter
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = polygon_perimeter([5, -1, 5, 10],
    ...                            [-1, 5, 11, 5],
    ...                            shape=img.shape, clip=True)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)

    """
    if clip:
        if shape is None:
            raise ValueError('Must specify clipping shape')
        clip_box = np.array([0, 0, shape[0] - 1, shape[1] - 1])
    else:
        clip_box = np.array([np.min(r), np.min(c), np.max(r), np.max(c)])
    r, c = polygon_clip(r, c, *clip_box)
    r = np.round(r).astype(int)
    c = np.round(c).astype(int)
    rr, cc = ([], [])
    for i in range(len(r) - 1):
        line_r, line_c = line(r[i], c[i], r[i + 1], c[i + 1])
        rr.extend(line_r)
        cc.extend(line_c)
    rr = np.asarray(rr)
    cc = np.asarray(cc)
    if shape is None:
        return (rr, cc)
    else:
        return _coords_inside_image(rr, cc, shape)