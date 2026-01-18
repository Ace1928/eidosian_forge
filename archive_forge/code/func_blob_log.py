import math
import numpy as np
import scipy.ndimage as ndi
from scipy import spatial
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max
def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False, *, threshold_rel=None, exclude_border=False):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        the minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float or None, optional
        The absolute lower bound for scale space maxima. Local maxima smaller
        than `threshold` are ignored. Reduce this to detect blobs with lower
        intensities. If `threshold_rel` is also specified, whichever threshold
        is larger will be used. If None, `threshold_rel` is used instead.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(log_space) * threshold_rel``, where ``log_space`` refers to the
        stack of Laplacian-of-Gaussian (LoG) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[124.        , 336.        ,  11.88888889],
           [198.        , 155.        ,  11.88888889],
           [194.        , 213.        ,  17.33333333],
           [121.        , 272.        ,  17.33333333],
           [263.        , 244.        ,  17.33333333],
           [194.        , 276.        ,  17.33333333],
           [266.        , 115.        ,  11.88888889],
           [128.        , 154.        ,  11.88888889],
           [260.        , 174.        ,  17.33333333],
           [198.        , 103.        ,  11.88888889],
           [126.        , 208.        ,  11.88888889],
           [127.        , 102.        ,  11.88888889],
           [263.        , 302.        ,  17.33333333],
           [197.        ,  44.        ,  11.88888889],
           [185.        , 344.        ,  17.33333333],
           [126.        ,  46.        ,  11.88888889],
           [113.        , 323.        ,   1.        ]])

    Notes
    -----
    The radius of each blob is approximately :math:`\\sqrt{2}\\sigma` for
    a 2-D image and :math:`\\sqrt{3}\\sigma` for a 3-D image.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    scalar_sigma = True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)
    if log_scale:
        start = np.log10(min_sigma)
        stop = np.log10(max_sigma)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    image_cube = np.empty(image.shape + (len(sigma_list),), dtype=float_dtype)
    for i, s in enumerate(sigma_list):
        image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s) ** 2
    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=exclude_border, footprint=np.ones((3,) * (image.ndim + 1)))
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))
    lm = local_maxima.astype(float_dtype)
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    if scalar_sigma:
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)