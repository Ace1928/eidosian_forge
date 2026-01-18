import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def structure_tensor(image, sigma=1, mode='constant', cval=0, order='rc'):
    """Compute structure tensor using sum of squared differences.

    The (2-dimensional) structure tensor A is defined as::

        A = [Arr Arc]
            [Arc Acc]

    which is approximated by the weighted sum of squared differences in a local
    window around each pixel in the image. This formula can be extended to a
    larger number of dimensions (see [1]_).

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float or array-like of float, optional
        Standard deviation used for the Gaussian kernel, which is used as a
        weighting function for the local summation of squared differences.
        If sigma is an iterable, its length must be equal to `image.ndim` and
        each element is used for the Gaussian kernel applied along its
        respective axis.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        NOTE: 'xy' is only an option for 2D images, higher dimensions must
        always use 'rc' order. This parameter allows for the use of reverse or
        forward order of the image axes in gradient computation. 'rc' indicates
        the use of the first axis initially (Arr, Arc, Acc), whilst 'xy'
        indicates the usage of the last axis initially (Axx, Axy, Ayy).

    Returns
    -------
    A_elems : list of ndarray
        Upper-diagonal elements of the structure tensor for each pixel in the
        input image.

    Examples
    --------
    >>> from skimage.feature import structure_tensor
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order='rc')
    >>> Acc
    array([[0., 0., 0., 0., 0.],
           [0., 1., 0., 1., 0.],
           [0., 4., 0., 4., 0.],
           [0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 0.]])

    See also
    --------
    structure_tensor_eigenvalues

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Structure_tensor
    """
    if order == 'xy' and image.ndim > 2:
        raise ValueError('Only "rc" order is supported for dim > 2.')
    if order not in ['rc', 'xy']:
        raise ValueError(f'order {order} is invalid. Must be either "rc" or "xy"')
    if not np.isscalar(sigma):
        sigma = tuple(sigma)
        if len(sigma) != image.ndim:
            raise ValueError('sigma must have as many elements as image has axes')
    image = _prepare_grayscale_input_nD(image)
    derivatives = _compute_derivatives(image, mode=mode, cval=cval)
    if order == 'xy':
        derivatives = reversed(derivatives)
    A_elems = [gaussian(der0 * der1, sigma=sigma, mode=mode, cval=cval) for der0, der1 in combinations_with_replacement(derivatives, 2)]
    return A_elems