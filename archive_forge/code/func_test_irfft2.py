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
def test_irfft2(self, xp):
    x = xp.asarray(random((30, 20)))
    xp_assert_close(fft.irfft2(fft.rfft2(x)), x)
    for norm in ['backward', 'ortho', 'forward']:
        xp_assert_close(fft.irfft2(fft.rfft2(x, norm=norm), norm=norm), x)