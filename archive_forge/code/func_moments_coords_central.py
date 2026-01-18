import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def moments_coords_central(coords, center=None, order=3):
    """Calculate all central image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space. A tuple of coordinates as returned by
        ``np.nonzero`` is also accepted as input.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    Mc : (``order + 1``, ``order + 1``, ...) array
        Central image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> coords = np.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)])
    >>> moments_coords_central(coords)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])

    As seen above, for symmetric objects, odd-order moments (columns 1 and 3,
    rows 1 and 3) are zero when centered on the centroid, or center of mass,
    of the object (the default). If we break the symmetry by adding a new
    point, this no longer holds:

    >>> coords2 = np.concatenate((coords, [[17, 17]]), axis=0)
    >>> np.round(moments_coords_central(coords2),
    ...          decimals=2)  # doctest: +NORMALIZE_WHITESPACE
    array([[17.  ,  0.  , 22.12, -2.49],
           [ 0.  ,  3.53,  1.73,  7.4 ],
           [25.88,  6.02, 36.63,  8.83],
           [ 4.15, 19.17, 14.8 , 39.6 ]])

    Image moments and central image moments are equivalent (by definition)
    when the center is (0, 0):

    >>> np.allclose(moments_coords(coords),
    ...             moments_coords_central(coords, (0, 0)))
    True
    """
    if isinstance(coords, tuple):
        coords = np.stack(coords, axis=-1)
    check_nD(coords, 2)
    ndim = coords.shape[1]
    float_type = _supported_float_type(coords.dtype)
    if center is None:
        center = np.mean(coords, axis=0, dtype=float)
    coords = coords.astype(float_type, copy=False) - center
    coords = np.stack([coords ** c for c in range(order + 1)], axis=-1)
    coords = coords.reshape(coords.shape + (1,) * (ndim - 1))
    calc = 1
    for axis in range(ndim):
        isolated_axis = coords[:, axis]
        isolated_axis = np.moveaxis(isolated_axis, 1, 1 + axis)
        calc = calc * isolated_axis
    Mc = np.sum(calc, axis=0)
    return Mc