import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
@pytest.mark.parametrize(['method', 'args'], window_funcs)
def test_deprecated_import(method, args):
    if method in ('taylor', 'lanczos', 'dpss'):
        pytest.skip('Deprecation test not applicable')
    func = getattr(signal, method)
    msg = f'Importing {method}'
    with pytest.deprecated_call(match=msg):
        func(1, *args)