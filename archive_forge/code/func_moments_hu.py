import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def moments_hu(nu):
    """Calculate Hu's set of image moments (2D-only).

    Note that this set of moments is proved to be translation, scale and
    rotation invariant.

    Parameters
    ----------
    nu : (M, M) array
        Normalized central image moments, where M must be >= 4.

    Returns
    -------
    nu : (7,) array
        Hu's set of image moments.

    References
    ----------
    .. [1] M. K. Hu, "Visual Pattern Recognition by Moment Invariants",
           IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
    .. [2] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [3] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [4] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [5] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.float64)
    >>> image[13:17, 13:17] = 0.5
    >>> image[10:12, 10:12] = 1
    >>> mu = moments_central(image)
    >>> nu = moments_normalized(mu)
    >>> moments_hu(nu)
    array([0.74537037, 0.35116598, 0.10404918, 0.04064421, 0.00264312,
           0.02408546, 0.        ])
    """
    dtype = np.float32 if nu.dtype == 'float32' else np.float64
    return _moments_cy.moments_hu(nu.astype(dtype, copy=False))