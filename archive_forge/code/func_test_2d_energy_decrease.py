import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert
def test_2d_energy_decrease():
    a_black = np.zeros((5, 5)).astype(np.uint8)
    a_black[2, 2] = 255
    a_white = invert(a_black)
    assert_array_less(meijering(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(), a_white.std())
    assert_array_less(sato(a_black, black_ridges=True, mode='reflect').std(), a_black.std())
    assert_array_less(sato(a_white, black_ridges=False, mode='reflect').std(), a_white.std())
    assert_array_less(frangi(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(), a_white.std())
    assert_array_less(hessian(a_black, black_ridges=True, mode='reflect').std(), a_black.std())
    assert_array_less(hessian(a_white, black_ridges=False, mode='reflect').std(), a_white.std())