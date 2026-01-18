from itertools import product
from numbers import Integral, Number, Real
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Hidden, Interval, RealNotInt, validate_params
@validate_params({'image': [np.ndarray], 'patch_size': [tuple, list], 'max_patches': [Interval(RealNotInt, 0, 1, closed='neither'), Interval(Integral, 1, None, closed='left'), None], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def extract_patches_2d(image, patch_size, *, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches.

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or         (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None it corresponds to the total number
        of patches that can be extracted.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or         (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the first image in this dataset:
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print('Patches shape: {}'.format(patches.shape))
    Patches shape: (272214, 2, 2, 3)
    >>> # Here are just two of these patches:
    >>> print(patches[1])
    [[[174 201 231]
      [174 201 231]]
     [[173 200 230]
      [173 200 230]]]
    >>> print(patches[800])
    [[[187 214 243]
      [188 215 244]]
     [[187 214 243]
      [188 215 244]]]
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size
    if p_h > i_h:
        raise ValueError('Height of the patch should be less than the height of the image.')
    if p_w > i_w:
        raise ValueError('Width of the patch should be less than the width of the image.')
    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    extracted_patches = _extract_patches(image, patch_shape=(p_h, p_w, n_colors), extraction_step=1)
    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches