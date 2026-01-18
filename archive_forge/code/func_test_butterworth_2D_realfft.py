import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
@pytest.mark.parametrize('high_pass', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D_realfft(high_pass, dtype, squared_butterworth):
    """Filtering a real-valued array is equivalent to filtering a
    complex-valued array where the imaginary part is zero.
    """
    im = np.random.randn(32, 64).astype(dtype)
    kwargs = dict(cutoff_frequency_ratio=0.2, high_pass=high_pass, squared_butterworth=squared_butterworth)
    expected_dtype = _supported_float_type(im.dtype)
    filtered_real = butterworth(im, **kwargs)
    assert filtered_real.dtype == expected_dtype
    cplx_dtype = np.promote_types(im.dtype, np.complex64)
    filtered_cplx = butterworth(im.astype(cplx_dtype), **kwargs)
    assert filtered_cplx.real.dtype == expected_dtype
    if expected_dtype == np.float64:
        rtol = atol = 1e-13
    else:
        rtol = atol = 1e-05
    assert_allclose(filtered_real, filtered_cplx.real, rtol=rtol, atol=atol)