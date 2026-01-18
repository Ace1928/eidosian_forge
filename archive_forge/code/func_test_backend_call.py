from functools import partial
import numpy as np
import scipy.fft
from scipy.fft import _fftlog, _pocketfft, set_backend
from scipy.fft.tests import mock_backend
from numpy.testing import assert_allclose, assert_equal
import pytest
@pytest.mark.parametrize('func, np_func, mock', zip(funcs, np_funcs, mocks))
def test_backend_call(func, np_func, mock):
    x = np.arange(20).reshape((10, 2))
    answer = np_func(x)
    assert_allclose(func(x), answer, atol=1e-10)
    with set_backend(mock_backend, only=True):
        mock.number_calls = 0
        y = func(x)
        assert_equal(y, mock.return_value)
        assert_equal(mock.number_calls, 1)
    assert_allclose(func(x), answer, atol=1e-10)