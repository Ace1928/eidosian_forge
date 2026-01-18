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
def fft1(x):
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)