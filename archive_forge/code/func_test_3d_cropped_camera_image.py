import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert
def test_3d_cropped_camera_image():
    a_black = crop(camera(), ((200, 212), (100, 312)))
    a_black = np.stack([a_black] * 5, axis=-1)
    a_white = invert(a_black)
    np.zeros(a_black.shape)
    ones = np.ones(a_black.shape)
    assert_allclose(meijering(a_black, black_ridges=True), meijering(a_white, black_ridges=False))
    assert_allclose(sato(a_black, black_ridges=True, mode='reflect'), sato(a_white, black_ridges=False, mode='reflect'))
    assert_allclose(frangi(a_black, black_ridges=True), frangi(a_white, black_ridges=False))
    assert_allclose(hessian(a_black, black_ridges=True, mode='reflect'), ones, atol=1 - 1e-07)
    assert_allclose(hessian(a_white, black_ridges=False, mode='reflect'), ones, atol=1 - 1e-07)