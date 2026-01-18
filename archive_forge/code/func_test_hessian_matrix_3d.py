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
def test_hessian_matrix_3d():
    cube = np.zeros((5, 5, 5))
    cube[2, 2, 2] = 4
    Hs = hessian_matrix(cube, sigma=0.1, order='rc', use_gaussian_derivatives=False)
    assert len(Hs) == 6, f'incorrect number of Hessian images ({len(Hs)}) for 3D'
    assert_almost_equal(Hs[2][:, 2, :], np.array([[0, 0, 0, 0, 0], [0, 1, 0, -1, 0], [0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0]]))
    assert_almost_equal(Hs[0][:, 2, :], np.array([[0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0]]))