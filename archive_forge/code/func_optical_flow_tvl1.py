from functools import partial
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian as gaussian_filter
from .._shared.utils import _supported_float_type
from ..transform import warp
from ._optical_flow_utils import _coarse_to_fine, _get_warp_points
def optical_flow_tvl1(reference_image, moving_image, *, attachment=15, tightness=0.3, num_warp=5, num_iter=10, tol=0.0001, prefilter=False, dtype=np.float32):
    """Coarse to fine optical flow estimator.

    The TV-L1 solver is applied at each level of the image
    pyramid. TV-L1 is a popular algorithm for optical flow estimation
    introduced by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first grayscale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second grayscale image of the sequence.
    attachment : float, optional
        Attachment parameter (:math:`\\lambda` in [1]_). The smaller
        this parameter is, the smoother the returned result will be.
    tightness : float, optional
        Tightness parameter (:math:`\\theta` in [1]_). It should have
        a small value in order to maintain attachment and
        regularization parts in correspondence.
    num_warp : int, optional
        Number of times moving_image is warped.
    num_iter : int, optional
        Number of fixed point iteration.
    tol : float, optional
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each
        image warp. When True, a median filter with window size 3
        along each axis is applied. This helps to remove potential
        outliers.
    dtype : dtype, optional
        Output data type: must be floating point. Single precision
        provides good results and saves memory usage and computation
        time compared to double precision.

    Returns
    -------
    flow : ndarray, shape (image0.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.

    Notes
    -----
    Color images are not supported.

    References
    ----------
    .. [1] Zach, C., Pock, T., & Bischof, H. (2007, September). A
       duality based approach for realtime TV-L 1 optical flow. In Joint
       pattern recognition symposium (pp. 214-223). Springer, Berlin,
       Heidelberg. :DOI:`10.1007/978-3-540-74936-3_22`
    .. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers,
       D. (2009). An improved algorithm for TV-L 1 optical flow. In
       Statistical and geometrical approaches to visual motion analysis
       (pp. 23-45). Springer, Berlin, Heidelberg.
       :DOI:`10.1007/978-3-642-03061-1_2`
    .. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo,
       G. (2013). TV-L1 optical flow estimation. Image Processing On
       Line, 2013, 137-150. :DOI:`10.5201/ipol.2013.26`

    Examples
    --------
    >>> from skimage.color import rgb2gray
    >>> from skimage.data import stereo_motorcycle
    >>> from skimage.registration import optical_flow_tvl1
    >>> image0, image1, disp = stereo_motorcycle()
    >>> # --- Convert the images to gray level: color is not supported.
    >>> image0 = rgb2gray(image0)
    >>> image1 = rgb2gray(image1)
    >>> flow = optical_flow_tvl1(image1, image0)

    """
    solver = partial(_tvl1, attachment=attachment, tightness=tightness, num_warp=num_warp, num_iter=num_iter, tol=tol, prefilter=prefilter)
    if np.dtype(dtype) != _supported_float_type(dtype):
        msg = f"dtype={dtype} is not supported. Try 'float32' or 'float64.'"
        raise ValueError(msg)
    return _coarse_to_fine(reference_image, moving_image, solver, dtype=dtype)