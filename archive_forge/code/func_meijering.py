from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals
def meijering(image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Meijering neuriteness filter.

    This filter can be used to detect continuous ridges, e.g. neurites,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvalues of the Hessian to compute the similarity of
    an image region to neurites, according to the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter
    alpha : float, optional
        Shaping filter constant, that selects maximally flat elongated
        features.  The default, None, selects the optimal value -1/(ndim+1).
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
    out : (M, N[, ...]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    sato
    frangi
    hessian

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
        Unser, M. (2004). Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images. Cytometry Part A,
        58(2), 167-176.
        :DOI:`10.1002/cyto.a.20022`
    """
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:
        image = -image
    if alpha is None:
        alpha = 1 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(hessian_matrix(image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        vals = np.tensordot(mtx, eigvals, 1)
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        vals = np.maximum(vals, 0)
        max_val = vals.max()
        if max_val > 0:
            vals /= max_val
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max