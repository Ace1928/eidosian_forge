import numpy as np
from scipy.stats import entropy
from ..util.dtype import dtype_range
from .._shared.utils import _supported_float_type, check_shape_equality, warn
def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    check_shape_equality(image_true, image_test)
    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn('Inputs have mismatched dtype.  Setting data_range based on image_true.')
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = (np.min(image_true), np.max(image_true))
        if true_max > dmax or true_min < dmin:
            raise ValueError('image_true has intensity values outside the range expected for its data type. Please manually specify the data_range.')
        if true_min >= 0:
            data_range = dmax
        else:
            data_range = dmax - dmin
    image_true, image_test = _as_floats(image_true, image_test)
    err = mean_squared_error(image_true, image_test)
    return 10 * np.log10(data_range ** 2 / err)