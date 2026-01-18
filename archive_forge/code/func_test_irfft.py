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
@array_api_compatible
def test_irfft(self, xp):
    a = xp.full(self.input_shape, 1 + 0j)
    self._test_mtsame(fft.irfft, a, xp=xp)