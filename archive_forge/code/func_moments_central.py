import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def moments_central(image, center=None, order=3, *, spacing=None, **kwargs):
    """Calculate all central image moments up to a certain order.

    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : (N[, ...]) double or uint8 array
        Rasterized shape as image.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        The maximum order of moments computed.
    spacing: tuple of float, shape (ndim,)
        The pixel spacing along each axis of the image.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.float64)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> moments_central(image, centroid)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    if center is None:
        moments_raw = moments(image, order=order, spacing=spacing)
        return moments_raw_to_central(moments_raw)
    float_dtype = _supported_float_type(image.dtype)
    if spacing is None:
        spacing = np.ones(image.ndim, dtype=float_dtype)
    calc = image.astype(float_dtype, copy=False)
    for dim, dim_length in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float_dtype) * spacing[dim] - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1, dtype=float_dtype)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc