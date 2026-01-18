import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, coins
from skimage.filters import (
@pytest.mark.parametrize('dtype', [np.uint8, np.float16, np.float32, np.float64])
def test_filter_inverse(self, dtype):
    img = self.img.astype(dtype, copy=False)
    expected_dtype = _supported_float_type(dtype)
    F = self.f(img)
    assert F.dtype == expected_dtype
    g = filter_inverse(F, predefined_filter=self.f)
    assert g.dtype == expected_dtype
    assert_equal(g.shape, self.img.shape)
    g1 = filter_inverse(F[::-1, ::-1], predefined_filter=self.f)
    assert_((g - g1[::-1, ::-1]).sum() < 55)
    g1 = filter_inverse(F[::-1, ::-1], predefined_filter=self.f)
    assert_((g - g1[::-1, ::-1]).sum() < 55)
    g1 = filter_inverse(F[::-1, ::-1], self.filt_func)
    assert_((g - g1[::-1, ::-1]).sum() < 55)