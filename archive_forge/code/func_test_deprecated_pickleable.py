import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_deprecated_pickleable():
    dep_hann2 = pickle.loads(pickle.dumps(dep_hann))
    assert_(dep_hann2 is dep_hann)