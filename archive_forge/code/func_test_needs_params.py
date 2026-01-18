import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_needs_params():
    for winstr in ['kaiser', 'ksr', 'kaiser_bessel_derived', 'kbd', 'gaussian', 'gauss', 'gss', 'general gaussian', 'general_gaussian', 'general gauss', 'general_gauss', 'ggs', 'dss', 'dpss', 'general cosine', 'general_cosine', 'chebwin', 'cheb', 'general hamming', 'general_hamming']:
        assert_raises(ValueError, get_window, winstr, 7)