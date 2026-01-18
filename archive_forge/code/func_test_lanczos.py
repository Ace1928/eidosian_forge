import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_lanczos(self):
    assert_allclose(get_window('lanczos', 6), [0.0, 0.413496672, 0.826993343, 1.0, 0.826993343, 0.413496672], atol=1e-09)
    assert_allclose(get_window('lanczos', 6, fftbins=False), [0.0, 0.504551152, 0.935489284, 0.935489284, 0.504551152, 0.0], atol=1e-09)
    assert_allclose(get_window('lanczos', 6), get_window('sinc', 6))