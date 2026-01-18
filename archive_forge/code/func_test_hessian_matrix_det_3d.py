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
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_hessian_matrix_det_3d(im3d, dtype):
    im3d = im3d.astype(dtype, copy=False)
    D = hessian_matrix_det(im3d)
    assert D.dtype == _supported_float_type(dtype)
    D0 = D[D.shape[0] // 2]
    row_center, col_center = np.array(D0.shape) // 2
    circles = [draw.circle_perimeter(row_center, col_center, r, shape=D0.shape) for r in range(1, D0.shape[1] // 2 - 1)]
    response = np.array([np.mean(D0[c]) for c in circles])
    lowest = np.argmin(response)
    highest = np.argmax(response)
    assert lowest < highest
    assert response[lowest] < 0
    assert response[highest] > 0