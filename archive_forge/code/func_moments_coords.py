import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def moments_coords(coords, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale, nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    M : (``order + 1``, ``order + 1``, ...) array
        Raw image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> coords = np.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)], dtype=np.float64)
    >>> M = moments_coords(coords)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> centroid
    (14.5, 15.5)
    """
    return moments_coords_central(coords, 0, order=order)