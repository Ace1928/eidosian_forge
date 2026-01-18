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
@skip_if_array_api_backend('torch')
def test_irfftn(self, xp):
    x = xp.asarray(random((30, 20, 10)))
    xp_assert_close(fft.irfftn(fft.rfftn(x)), x)
    for norm in ['backward', 'ortho', 'forward']:
        xp_assert_close(fft.irfftn(fft.rfftn(x, norm=norm), norm=norm), x)