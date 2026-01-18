import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
def test_bad_size_zi(self):
    x1 = np.arange(6)
    self.base_bad_size_zi([1], [1], x1, -1, [1])
    self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1])
    self.base_bad_size_zi([1, 1], [1], x1, -1, [[0]])
    self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1, 2])
    self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [[0]])
    self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [0, 1, 2])
    self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1])
    self.base_bad_size_zi([1], [1, 1], x1, -1, [[0]])
    self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1, 2])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [[0], [1]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2, 3])
    self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0])
    self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [[0], [1]])
    self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2])
    self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2, 3])
    x2 = np.arange(12).reshape((4, 3))
    self.base_bad_size_zi([1], [1], x2, 0, [0])
    self.base_bad_size_zi([1, 1], [1], x2, 0, [0, 1, 2])
    self.base_bad_size_zi([1, 1], [1], x2, 0, [[[0, 1, 2]]])
    self.base_bad_size_zi([1, 1], [1], x2, 0, [[0], [1], [2]])
    self.base_bad_size_zi([1, 1], [1], x2, 0, [[0, 1]])
    self.base_bad_size_zi([1, 1], [1], x2, 0, [[0, 1, 2, 3]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [0, 1, 2, 3, 4, 5])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1], [2, 3]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1], [1, 1], x2, 0, [0, 1, 2])
    self.base_bad_size_zi([1], [1, 1], x2, 0, [[[0, 1, 2]]])
    self.base_bad_size_zi([1], [1, 1], x2, 0, [[0], [1], [2]])
    self.base_bad_size_zi([1], [1, 1], x2, 0, [[0, 1]])
    self.base_bad_size_zi([1], [1, 1], x2, 0, [[0, 1, 2, 3]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [0, 1, 2, 3, 4, 5])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1], [2, 3]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [0, 1, 2, 3, 4, 5])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[[0, 1, 2], [3, 4, 5]]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1], [2, 3]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1], [1], x2, 1, [0])
    self.base_bad_size_zi([1, 1], [1], x2, 1, [0, 1, 2, 3])
    self.base_bad_size_zi([1, 1], [1], x2, 1, [[[0], [1], [2], [3]]])
    self.base_bad_size_zi([1, 1], [1], x2, 1, [[0, 1, 2, 3]])
    self.base_bad_size_zi([1, 1], [1], x2, 1, [[0], [1], [2]])
    self.base_bad_size_zi([1, 1], [1], x2, 1, [[0], [1], [2], [3], [4]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    self.base_bad_size_zi([1], [1, 1], x2, 1, [0, 1, 2, 3])
    self.base_bad_size_zi([1], [1, 1], x2, 1, [[[0], [1], [2], [3]]])
    self.base_bad_size_zi([1], [1, 1], x2, 1, [[0, 1, 2, 3]])
    self.base_bad_size_zi([1], [1, 1], x2, 1, [[0], [1], [2]])
    self.base_bad_size_zi([1], [1, 1], x2, 1, [[0], [1], [2], [3], [4]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [0, 1, 2, 3, 4, 5, 6, 7])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[[0, 1], [2, 3], [4, 5], [6, 7]]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1], [2, 3], [4, 5]])
    self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])