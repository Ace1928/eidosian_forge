import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.uint8, np.int32])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D_zeros_dtypes(dtype, squared_butterworth):
    im = np.zeros((4, 4), dtype=dtype)
    filtered = butterworth(im, squared_butterworth=squared_butterworth)
    assert filtered.shape == im.shape
    assert filtered.dtype == _supported_float_type(dtype)
    assert_array_equal(im, filtered)