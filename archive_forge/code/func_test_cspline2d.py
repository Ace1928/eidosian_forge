import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_cspline2d():
    np.random.seed(181819142)
    image = np.random.rand(71, 73)
    signal.cspline2d(image, 8.0)