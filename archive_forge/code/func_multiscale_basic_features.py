from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage.util.dtype import img_as_float32
from concurrent.futures import ThreadPoolExecutor
def multiscale_basic_features(image, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16, num_sigma=None, num_workers=None, *, channel_axis=None):
    """Local features for a single- or multi-channel nd image.

    Intensity, gradient intensity and local structure are computed at
    different scales thanks to Gaussian blurring.

    Parameters
    ----------
    image : ndarray
        Input image, which can be grayscale or multichannel.
    intensity : bool, default True
        If True, pixel intensities averaged over the different scales
        are added to the feature set.
    edges : bool, default True
        If True, intensities of local gradients averaged over the different
        scales are added to the feature set.
    texture : bool, default True
        If True, eigenvalues of the Hessian matrix after Gaussian blurring
        at different scales are added to the feature set.
    sigma_min : float, optional
        Smallest value of the Gaussian kernel used to average local
        neighborhoods before extracting features.
    sigma_max : float, optional
        Largest value of the Gaussian kernel used to average local
        neighborhoods before extracting features.
    num_sigma : int, optional
        Number of values of the Gaussian kernel between sigma_min and sigma_max.
        If None, sigma_min multiplied by powers of 2 are used.
    num_workers : int or None, optional
        The number of parallel threads to use. If set to ``None``, the full
        set of available cores are used.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    features : np.ndarray
        Array of shape ``image.shape + (n_features,)``. When `channel_axis` is
        not None, all channels are concatenated along the features dimension.
        (i.e. ``n_features == n_features_singlechannel * n_channels``)
    """
    if not any([intensity, edges, texture]):
        raise ValueError('At least one of `intensity`, `edges` or `textures`must be True for features to be computed.')
    if channel_axis is None:
        image = image[..., np.newaxis]
        channel_axis = -1
    elif channel_axis != -1:
        image = np.moveaxis(image, channel_axis, -1)
    all_results = (_mutiscale_basic_features_singlechannel(image[..., dim], intensity=intensity, edges=edges, texture=texture, sigma_min=sigma_min, sigma_max=sigma_max, num_sigma=num_sigma, num_workers=num_workers) for dim in range(image.shape[-1]))
    features = list(itertools.chain.from_iterable(all_results))
    out = np.stack(features, axis=-1)
    return out