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
def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False, *, threshold_rel=None):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
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
        ``max(doh_space) * threshold_rel``, where ``doh_space`` refers to the
        stack of Determinant-of-Hessian (DoH) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian
    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Examples
    --------
    >>> from skimage import data, feature
    >>> img = data.coins()
    >>> feature.blob_doh(img)
    array([[197.        , 153.        ,  20.33333333],
           [124.        , 336.        ,  20.33333333],
           [126.        , 153.        ,  20.33333333],
           [195.        , 100.        ,  23.55555556],
           [192.        , 212.        ,  23.55555556],
           [121.        , 271.        ,  30.        ],
           [126.        , 101.        ,  20.33333333],
           [193.        , 275.        ,  23.55555556],
           [123.        , 205.        ,  20.33333333],
           [270.        , 363.        ,  30.        ],
           [265.        , 113.        ,  23.55555556],
           [262.        , 243.        ,  23.55555556],
           [185.        , 348.        ,  30.        ],
           [156.        , 302.        ,  30.        ],
           [123.        ,  44.        ,  23.55555556],
           [260.        , 173.        ,  30.        ],
           [197.        ,  44.        ,  20.33333333]])

    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """
    check_nD(image, 2)
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = integral_image(image)
    if log_scale:
        start, stop = (math.log(min_sigma, 10), math.log(max_sigma, 10))
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    image_cube = np.empty(shape=image.shape + (len(sigma_list),), dtype=float_dtype)
    for j, s in enumerate(sigma_list):
        image_cube[..., j] = _hessian_matrix_det(image, s)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=False, footprint=np.ones((3,) * image_cube.ndim))
    if local_maxima.size == 0:
        return np.empty((0, 3))
    lm = local_maxima.astype(np.float64)
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)