import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
def test_apply_parallel():
    a = np.arange(144).reshape(12, 12).astype(float)
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5, extra_arguments=(3,), extra_keywords={'mode': 'reflect'})
    assert_array_almost_equal(result1, expected1)

    def wrapped_gauss(arr):
        return gaussian(arr, sigma=1, mode='reflect')
    expected2 = gaussian(a, sigma=1, mode='reflect')
    result2 = apply_parallel(wrapped_gauss, a, chunks=(6, 6), depth=5)
    assert_array_almost_equal(result2, expected2)
    expected3 = gaussian(a, sigma=1, mode='reflect')
    result3 = apply_parallel(wrapped_gauss, da.from_array(a, chunks=(6, 6)), depth=5, compute=True)
    assert isinstance(result3, np.ndarray)
    assert_array_almost_equal(result3, expected3)