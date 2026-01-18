import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, coins
from skimage.filters import (
def test_filter_forward():

    def filt_func(r, c, sigma=2):
        return 1 / (2 * np.pi * sigma ** 2) * np.exp(-(r ** 2 + c ** 2) / (2 * sigma ** 2))
    gaussian_args = {'sigma': 2, 'preserve_range': True, 'mode': 'constant', 'truncate': 20}
    image = coins()[:303, :383]
    filtered = filter_forward(image, filt_func)
    filtered_gaussian = gaussian(image, **gaussian_args)
    assert_array_almost_equal(filtered, filtered_gaussian)
    image = coins()
    filtered = filter_forward(image, filt_func)
    filtered_gaussian = gaussian(image, **gaussian_args)
    assert_array_almost_equal(filtered, filtered_gaussian)