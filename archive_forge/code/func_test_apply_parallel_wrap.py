import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
def test_apply_parallel_wrap():

    def wrapped(arr):
        return gaussian(arr, sigma=1, mode='wrap')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian(a, sigma=1, mode='wrap')
    result = apply_parallel(wrapped, a, chunks=(6, 6), depth=5, mode='wrap')
    assert_array_almost_equal(result, expected)