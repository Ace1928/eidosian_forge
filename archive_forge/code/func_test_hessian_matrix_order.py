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
def test_hessian_matrix_order(use_gaussian_derivatives):
    square = np.zeros((5, 5), dtype=float)
    square[2, 2] = 4
    Hxx, Hxy, Hyy = hessian_matrix(square, sigma=0.1, order='xy', use_gaussian_derivatives=use_gaussian_derivatives)
    Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc', use_gaussian_derivatives=use_gaussian_derivatives)
    assert_array_equal(Hxx, Hcc)
    assert_array_equal(Hxy, Hrc)
    assert_array_equal(Hyy, Hrr)