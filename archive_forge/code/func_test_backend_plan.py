from functools import partial
import numpy as np
import scipy.fft
from scipy.fft import _fftlog, _pocketfft, set_backend
from scipy.fft.tests import mock_backend
from numpy.testing import assert_allclose, assert_equal
import pytest
@pytest.mark.parametrize('func, mock', zip(plan_funcs, plan_mocks))
def test_backend_plan(func, mock):
    x = np.arange(20).reshape((10, 2))
    with pytest.raises(NotImplementedError, match='precomputed plan'):
        func(x, plan='foo')
    with set_backend(mock_backend, only=True):
        mock.number_calls = 0
        y = func(x, plan='foo')
        assert_equal(y, mock.return_value)
        assert_equal(mock.number_calls, 1)
        assert_equal(mock.last_args[1]['plan'], 'foo')