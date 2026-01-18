import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_too_large_with_multiindex():
    base_size = 2 ** 10
    num = 1
    while base_size ** num < np.iinfo(np.intp).max:
        num += 1
    shape_template = [1, 1] * num
    arrays = []
    for i in range(num):
        shape = shape_template[:]
        shape[i * 2] = 2 ** 10
        arrays.append(np.empty(shape))
    arrays = tuple(arrays)
    for mode in range(6):
        with assert_raises(ValueError):
            _multiarray_tests.test_nditer_too_large(arrays, -1, mode)
    _multiarray_tests.test_nditer_too_large(arrays, -1, 7)
    for i in range(num):
        for mode in range(6):
            _multiarray_tests.test_nditer_too_large(arrays, i * 2, mode)
            with assert_raises(ValueError):
                _multiarray_tests.test_nditer_too_large(arrays, i * 2 + 1, mode)