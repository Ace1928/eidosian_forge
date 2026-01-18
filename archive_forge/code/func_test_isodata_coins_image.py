import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
def test_isodata_coins_image():
    coins = util.img_as_ubyte(data.coins())
    threshold = threshold_isodata(coins)
    assert np.floor((coins[coins <= threshold].mean() + coins[coins > threshold].mean()) / 2.0) == threshold
    assert threshold == 107
    assert threshold_isodata(coins, return_all=True) == [107]