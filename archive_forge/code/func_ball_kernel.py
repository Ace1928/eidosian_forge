import numpy as np
from .._shared.utils import _supported_float_type
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan
def ball_kernel(radius, ndim):
    """Create a ball kernel for restoration.rolling_ball.

    Parameters
    ----------
    radius : int
        Radius of the ball.
    ndim : int
        Number of dimensions of the ball. ``ndim`` should match the
        dimensionality of the image the kernel will be applied to.

    Returns
    -------
    kernel : ndarray
        The kernel containing the surface intensity of the top half
        of the ellipsoid.

    See Also
    --------
    rolling_ball
    """
    kernel_coords = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in [np.ceil(radius)] * ndim], indexing='ij'), axis=-1)
    sum_of_squares = np.sum(kernel_coords ** 2, axis=-1)
    distance_from_center = np.sqrt(sum_of_squares)
    kernel = np.sqrt(np.clip(radius ** 2 - sum_of_squares, 0, None))
    kernel[distance_from_center > radius] = np.inf
    return kernel