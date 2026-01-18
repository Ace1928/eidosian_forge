from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from ..measure.pnpoly import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows
from .._shared.utils import warn
def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10, include_borders=True):
    """Compute the convex hull image of a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : array
        Binary input image. This array is cast to bool before processing.
    offset_coordinates : bool, optional
        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented
        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds
        some "extent" to a pixel when computing the hull.
    tolerance : float, optional
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.
    include_borders: bool, optional
        If ``False``, vertices/edges are excluded from the final hull mask.

    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.

    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
    ndim = image.ndim
    if np.count_nonzero(image) == 0:
        warn('Input image is entirely zero, no valid convex hull. Returning empty image', UserWarning)
        return np.zeros(image.shape, dtype=bool)
    if ndim == 2:
        coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
    else:
        coords = np.transpose(np.nonzero(image))
        if offset_coordinates:
            try:
                hull0 = ConvexHull(coords)
            except QhullError as err:
                warn(f'Failed to get convex hull image. Returning empty image, see error message below:\n{err}')
                return np.zeros(image.shape, dtype=bool)
            coords = hull0.points[hull0.vertices]
    if offset_coordinates:
        offsets = _offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)
    coords = unique_rows(coords)
    try:
        hull = ConvexHull(coords)
    except QhullError as err:
        warn(f'Failed to get convex hull image. Returning empty image, see error message below:\n{err}')
        return np.zeros(image.shape, dtype=bool)
    vertices = hull.points[hull.vertices]
    if ndim == 2:
        labels = grid_points_in_poly(image.shape, vertices, binarize=False)
        mask = labels >= 1 if include_borders else labels == 1
    else:
        gridcoords = np.reshape(np.mgrid[tuple(map(slice, image.shape))], (ndim, -1))
        coords_in_hull = _check_coords_in_hull(gridcoords, hull.equations, tolerance)
        mask = np.reshape(coords_in_hull, image.shape)
    return mask