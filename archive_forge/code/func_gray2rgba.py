from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def gray2rgba(image, alpha=None, *, channel_axis=-1):
    """Create a RGBA representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    alpha : array_like, optional
        Alpha channel of the output image. It may be a scalar or an
        array that can be broadcast to ``image``. If not specified it is
        set to the maximum limit corresponding to the ``image`` dtype.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    rgba : ndarray
        RGBA image. A new dimension of length 4 is added to input
        image shape.
    """
    arr = np.asarray(image)
    if alpha is None:
        _, alpha = dtype_limits(arr, clip_negative=False)
    with np.errstate(over='ignore', under='ignore'):
        alpha_arr = np.asarray(alpha).astype(arr.dtype)
    if not np.array_equal(alpha_arr, alpha):
        warn(f'alpha cannot be safely cast to image dtype {arr.dtype.name}', stacklevel=2)
    try:
        alpha_arr = np.broadcast_to(alpha_arr, arr.shape)
    except ValueError as e:
        raise ValueError('alpha.shape must match image.shape') from e
    rgba = np.stack((arr,) * 3 + (alpha_arr,), axis=channel_axis)
    return rgba