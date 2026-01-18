import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
def denoise_invariant(image, denoise_function, *, stride=4, masks=None, denoiser_kwargs=None):
    """Apply a J-invariant version of a denoising function.

    Parameters
    ----------
    image : ndarray (M[, N[, ...]][, C]) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into a ndarray of floats (using `img_as_float`) for the
        computation of the denoised image.
    denoise_function : function
        Original denoising function.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    masks : list of ndarray, optional
        Set of masks to use for computing J-invariant output. If `None`,
        a full set of masks covering the image will be used.
    denoiser_kwargs:
        Keyword arguments passed to `denoise_function`.

    Returns
    -------
    output : ndarray
        Denoised image, of same shape as `image`.

    Notes
    -----
    A denoising function is J-invariant if the prediction it makes for each
    pixel does not depend on the value of that pixel in the original image.
    The prediction for each pixel may instead use all the relevant information
    contained in the rest of the image, which is typically quite significant.
    Any function can be converted into a J-invariant one using a simple masking
    procedure, as described in [1].

    The pixel-wise error of a J-invariant denoiser is uncorrelated to the noise,
    so long as the noise in each pixel is independent. Consequently, the average
    difference between the denoised image and the oisy image, the
    *self-supervised loss*, is the same as the difference between the denoised
    image and the original clean image, the *ground-truth loss* (up to a
    constant).

    This means that the best J-invariant denoiser for a given image can be found
    using the noisy data alone, by selecting the denoiser minimizing the self-
    supervised loss.

    References
    ----------
    .. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,
       International Conference on Machine Learning, p. 524-533 (2019).

    Examples
    --------
    >>> import skimage
    >>> from skimage.restoration import denoise_invariant, denoise_tv_chambolle
    >>> image = skimage.util.img_as_float(skimage.data.chelsea())
    >>> noisy = skimage.util.random_noise(image, var=0.2 ** 2)
    >>> denoised = denoise_invariant(noisy, denoise_function=denoise_tv_chambolle)
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if denoiser_kwargs is None:
        denoiser_kwargs = {}
    multichannel = denoiser_kwargs.get('channel_axis', None) is not None
    interp = _interpolate_image(image, multichannel=multichannel)
    output = np.zeros_like(image)
    if masks is None:
        spatialdims = image.ndim if not multichannel else image.ndim - 1
        n_masks = stride ** spatialdims
        masks = (_generate_grid_slice(image.shape[:spatialdims], offset=idx, stride=stride) for idx in range(n_masks))
    for mask in masks:
        input_image = image.copy()
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]
    return output