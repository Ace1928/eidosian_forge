import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_too_large():
    size = np.iinfo(np.intp).max // 1024
    arr = np.lib.stride_tricks.as_strided(np.zeros(1), (size,), (0,))
    assert_raises(ValueError, nditer, (arr, arr[:, None]))
    assert_raises(ValueError, nditer, (arr, arr[:, None]), flags=['multi_index'])