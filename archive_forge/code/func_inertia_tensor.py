import itertools
import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
from ._moments_analytical import moments_raw_to_central
def inertia_tensor(image, mu=None, *, spacing=None):
    """Compute the inertia tensor of the input image.

    Parameters
    ----------
    image : array
        The input image.
    mu : array, optional
        The pre-computed central moments of ``image``. The inertia tensor
        computation requires the central moments of the image. If an
        application requires both the central moments and the inertia tensor
        (for example, `skimage.measure.regionprops`), then it is more
        efficient to pre-compute them and pass them to the inertia tensor
        call.
    spacing: tuple of float, shape (ndim,)
        The pixel spacing along each axis of the image.

    Returns
    -------
    T : array, shape ``(image.ndim, image.ndim)``
        The inertia tensor of the input image. :math:`T_{i, j}` contains
        the covariance of image intensity along axes :math:`i` and :math:`j`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    .. [2] Bernd Jähne. Spatio-Temporal Image Processing: Theory and
           Scientific Applications. (Chapter 8: Tensor Methods) Springer, 1993.
    """
    if mu is None:
        mu = moments_central(image, order=2, spacing=spacing)
    mu0 = mu[(0,) * image.ndim]
    result = np.zeros((image.ndim, image.ndim), dtype=mu.dtype)
    corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
    d = np.diag(result)
    d.flags.writeable = True
    d[:] = (np.sum(mu[corners2]) - mu[corners2]) / mu0
    for dims in itertools.combinations(range(image.ndim), 2):
        mu_index = np.zeros(image.ndim, dtype=int)
        mu_index[list(dims)] = 1
        result[dims] = -mu[tuple(mu_index)] / mu0
        result.T[dims] = -mu[tuple(mu_index)] / mu0
    return result