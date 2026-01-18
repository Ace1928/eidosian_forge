import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
def calibrate_denoiser(image, denoise_function, denoise_parameters, *, stride=4, approximate_loss=True, extra_output=False):
    """Calibrate a denoising function and return optimal J-invariant version.

    The returned function is partially evaluated with optimal parameter values
    set for denoising the input image.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    approximate_loss : bool, optional
        Whether to approximate the self-supervised loss used to evaluate the
        denoiser by only computing it on one masked version of the image.
        If False, the runtime will be a factor of `stride**image.ndim` longer.
    extra_output : bool, optional
        If True, return parameters and losses in addition to the calibrated
        denoising function

    Returns
    -------
    best_denoise_function : function
        The optimal J-invariant version of `denoise_function`.

    If `extra_output` is True, the following tuple is also returned:

    (parameters_tested, losses) : tuple (list of dict, list of int)
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs
        Self-supervised loss for each set of parameters in `parameters_tested`.


    Notes
    -----

    The calibration procedure uses a self-supervised mean-square-error loss
    to evaluate the performance of J-invariant versions of `denoise_function`.
    The minimizer of the self-supervised loss is also the minimizer of the
    ground-truth loss (i.e., the true MSE error) [1]. The returned function
    can be used on the original noisy image, or other images with similar
    characteristics.

    Increasing the stride increases the performance of `best_denoise_function`
     at the expense of increasing its runtime. It has no effect on the runtime
     of the calibration.

    References
    ----------
    .. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,
           International Conference on Machine Learning, p. 524-533 (2019).

    Examples
    --------
    >>> from skimage import color, data
    >>> from skimage.restoration import denoise_tv_chambolle
    >>> import numpy as np
    >>> img = color.rgb2gray(data.astronaut()[:50, :50])
    >>> rng = np.random.default_rng()
    >>> noisy = img + 0.5 * img.std() * rng.standard_normal(img.shape)
    >>> parameters = {'weight': np.arange(0.01, 0.3, 0.02)}
    >>> denoising_function = calibrate_denoiser(noisy, denoise_tv_chambolle,
    ...                                         denoise_parameters=parameters)
    >>> denoised_img = denoising_function(img)

    """
    parameters_tested, losses = _calibrate_denoiser_search(image, denoise_function, denoise_parameters=denoise_parameters, stride=stride, approximate_loss=approximate_loss)
    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]
    best_denoise_function = functools.partial(denoise_invariant, denoise_function=denoise_function, stride=stride, denoiser_kwargs=best_parameters)
    if extra_output:
        return (best_denoise_function, (parameters_tested, losses))
    else:
        return best_denoise_function