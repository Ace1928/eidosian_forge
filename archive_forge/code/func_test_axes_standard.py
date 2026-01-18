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
@pytest.mark.parametrize('op', [fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn])
def test_axes_standard(self, op, xp):
    self._check_axes(op, xp)