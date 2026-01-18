import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
@skip_if_array_api_backend('torch')
@array_api_compatible
@pytest.mark.parametrize('func', [fft.fft, fft.ifft, fft.rfft, fft.irfft, fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn, fft.hfft, fft.ihfft])
def test_non_standard_params(func, xp):
    if xp.__name__ != 'numpy':
        x = xp.asarray([1, 2, 3])
        func(x)
        assert_raises(ValueError, func, x, workers=2)