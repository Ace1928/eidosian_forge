from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('func', [rfftn, irfftn])
def test_invalid_sizes(self, func):
    with assert_raises(ValueError, match='invalid number of data points \\(\\[1, 0\\]\\) specified'):
        func([[]])
    with assert_raises(ValueError, match='invalid number of data points \\(\\[4, -3\\]\\) specified'):
        func([[1, 1], [2, 2]], (4, -3))