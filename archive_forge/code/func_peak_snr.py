import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
def peak_snr(img1, img2):
    """Peak signal to noise ratio of two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    peak_snr : float
        Peak signal to noise ratio
    """
    if img1.ndim == 3:
        img1, img2 = (rgb2gray(img1.copy()), rgb2gray(img2.copy()))
    img1 = util.img_as_float(img1)
    img2 = util.img_as_float(img2)
    mse = 1.0 / img1.size * np.square(img1 - img2).sum()
    _, max_ = dtype_range[img1.dtype.type]
    return 20 * np.log(max_ / mse)