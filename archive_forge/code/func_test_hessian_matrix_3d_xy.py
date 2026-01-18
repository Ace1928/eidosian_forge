import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
@pytest.mark.parametrize('use_gaussian_derivatives', [False, True])
def test_hessian_matrix_3d_xy(use_gaussian_derivatives):
    img = np.ones((5, 5, 5))
    with pytest.raises(ValueError):
        hessian_matrix(img, sigma=0.1, order='xy', use_gaussian_derivatives=use_gaussian_derivatives)
    with pytest.raises(ValueError):
        hessian_matrix(img, sigma=0.1, order='nonexistant', use_gaussian_derivatives=use_gaussian_derivatives)