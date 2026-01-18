from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals
def frangi(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=None, black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
        The default, None, uses half of the maximum Hessian norm.
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

    Notes
    -----
    Earlier versions of this filter were implemented by Marc Schrijver,
    (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and
    D. G. Ellis (January 2017) [3]_.

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn('Use keyword parameter `sigmas` instead of `scale_range` and `scale_range` which will be removed in version 0.17.', stacklevel=2)
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    check_nD(image, [2, 3])
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        lambda1 = eigvals[0]
        if image.ndim == 2:
            lambda2, = np.maximum(eigvals[1:], 1e-10)
            r_a = np.inf
            r_b = abs(lambda1) / lambda2
        else:
            lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
            r_a = lambda2 / lambda3
            r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)
        s = np.sqrt((eigvals ** 2).sum(0))
        if gamma is None:
            gamma = s.max() / 2
            if gamma == 0:
                gamma = 1
        vals = 1.0 - np.exp(-r_a ** 2 / (2 * alpha ** 2), dtype=image.dtype)
        vals *= np.exp(-r_b ** 2 / (2 * beta ** 2), dtype=image.dtype)
        vals *= 1.0 - np.exp(-s ** 2 / (2 * gamma ** 2), dtype=image.dtype)
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max