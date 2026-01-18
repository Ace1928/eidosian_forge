from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('func', [rfftn, irfftn])
def test_no_axes(self, func):
    with assert_raises(ValueError, match='at least 1 axis must be transformed'):
        func([], axes=[])