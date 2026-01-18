import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert
def test_2d_linearity():
    a_black = np.ones((3, 3)).astype(np.uint8)
    a_white = invert(a_black)
    assert_allclose(meijering(1 * a_black, black_ridges=True), meijering(10 * a_black, black_ridges=True), atol=0.001)
    assert_allclose(meijering(1 * a_white, black_ridges=False), meijering(10 * a_white, black_ridges=False), atol=0.001)
    assert_allclose(sato(1 * a_black, black_ridges=True, mode='reflect'), sato(10 * a_black, black_ridges=True, mode='reflect'), atol=0.001)
    assert_allclose(sato(1 * a_white, black_ridges=False, mode='reflect'), sato(10 * a_white, black_ridges=False, mode='reflect'), atol=0.001)
    assert_allclose(frangi(1 * a_black, black_ridges=True), frangi(10 * a_black, black_ridges=True), atol=0.001)
    assert_allclose(frangi(1 * a_white, black_ridges=False), frangi(10 * a_white, black_ridges=False), atol=0.001)
    assert_allclose(hessian(1 * a_black, black_ridges=True, mode='reflect'), hessian(10 * a_black, black_ridges=True, mode='reflect'), atol=0.001)
    assert_allclose(hessian(1 * a_white, black_ridges=False, mode='reflect'), hessian(10 * a_white, black_ridges=False, mode='reflect'), atol=0.001)