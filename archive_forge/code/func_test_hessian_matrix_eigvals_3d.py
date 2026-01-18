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
def test_hessian_matrix_eigvals_3d(im3d, dtype):
    im3d = im3d.astype(dtype, copy=False)
    H = hessian_matrix(im3d, use_gaussian_derivatives=False)
    E = hessian_matrix_eigvals(H)
    out_dtype = _supported_float_type(dtype)
    assert all((a.dtype == out_dtype for a in E))
    e0, e1, e2 = E
    assert np.all(e0 >= e1) and np.all(e1 >= e2)
    E0, E1, E2 = E[:, E.shape[1] // 2]
    row_center, col_center = np.array(E0.shape) // 2
    circles = [draw.circle_perimeter(row_center, col_center, radius, shape=E0.shape) for radius in range(1, E0.shape[1] // 2 - 1)]
    response0 = np.array([np.mean(E0[c]) for c in circles])
    response2 = np.array([np.mean(E2[c]) for c in circles])
    assert np.argmin(response2) < np.argmax(response0)
    assert np.min(response2) < 0
    assert np.max(response0) > 0