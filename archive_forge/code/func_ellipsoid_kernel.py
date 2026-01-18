import numpy as np
from .._shared.utils import _supported_float_type
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan
def ellipsoid_kernel(shape, intensity):
    """Create an ellipoid kernel for restoration.rolling_ball.

    Parameters
    ----------
    shape : array-like
        Length of the principal axis of the ellipsoid (excluding
        the intensity axis). The kernel needs to have the same
        dimensionality as the image it will be applied to.
    intensity : int
        Length of the intensity axis of the ellipsoid.

    Returns
    -------
    kernel : ndarray
        The kernel containing the surface intensity of the top half
        of the ellipsoid.

    See Also
    --------
    rolling_ball
    """
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None)
    kernel_coords = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in semi_axis], indexing='ij'), axis=-1)
    intensity_scaling = 1 - np.sum((kernel_coords / semi_axis) ** 2, axis=-1)
    kernel = intensity * np.sqrt(np.clip(intensity_scaling, 0, None))
    kernel[intensity_scaling < 0] = np.inf
    return kernel