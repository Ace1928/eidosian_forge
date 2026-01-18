import numpy as np
from ..util import crop
from ._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance
from ._util import (
from .._shared.dtype import numeric_dtype_min_max
def flood_fill(image, seed_point, new_value, *, footprint=None, connectivity=None, tolerance=None, in_place=False):
    """Perform flood filling on an image.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found, then set to `new_value`.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    new_value : `image` type
        New value to set the entire fill.  This must be chosen in agreement
        with the dtype of `image`.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`. If not given, all adjacent pixels
        are considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        value of `image` at `seed_point` to be filled.  This is fastest.
        If a tolerance is provided, adjacent points with values within plus or
        minus tolerance from the seed point are filled (inclusive).
    in_place : bool, optional
        If True, flood filling is applied to `image` in place.  If False, the
        flood filled result is returned without modifying the input `image`
        (default).

    Returns
    -------
    filled : ndarray
        An array with the same shape as `image` is returned, with values in
        areas connected to and equal (or within tolerance of) the seed point
        replaced with `new_value`.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.

    Examples
    --------
    >>> from skimage.morphology import flood_fill
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> flood_fill(image, (1, 1), 5)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> flood_fill(image, (1, 1), 5, connectivity=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> flood_fill(image, (0, 0), 5, tolerance=1)
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    mask = flood(image, seed_point, footprint=footprint, connectivity=connectivity, tolerance=tolerance)
    if not in_place:
        image = image.copy()
    image[mask] = new_value
    return image