from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals
def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to tubes, according to
    the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    meijering
    frangi
    hessian

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
        :DOI:`10.1016/S1361-8415(98)80009-1`
    """
    check_nD(image, [2, 3])
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        eigvals = eigvals[:-1]
        vals = sigma ** 2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max