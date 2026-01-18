import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_int64_uint64_broadcast_exceptions(self, endpoint):
    configs = {np.uint64: ((0, 2 ** 65), (-1, 2 ** 62), (10, 9), (0, 0)), np.int64: ((0, 2 ** 64), (-2 ** 64, 2 ** 62), (10, 9), (0, 0), (-2 ** 63 - 1, -2 ** 63 - 1))}
    for dtype in configs:
        for config in configs[dtype]:
            low, high = config
            high = high - endpoint
            low_a = np.array([[low] * 10])
            high_a = np.array([high] * 10)
            assert_raises(ValueError, random.integers, low, high, endpoint=endpoint, dtype=dtype)
            assert_raises(ValueError, random.integers, low_a, high, endpoint=endpoint, dtype=dtype)
            assert_raises(ValueError, random.integers, low, high_a, endpoint=endpoint, dtype=dtype)
            assert_raises(ValueError, random.integers, low_a, high_a, endpoint=endpoint, dtype=dtype)
            low_o = np.array([[low] * 10], dtype=object)
            high_o = np.array([high] * 10, dtype=object)
            assert_raises(ValueError, random.integers, low_o, high, endpoint=endpoint, dtype=dtype)
            assert_raises(ValueError, random.integers, low, high_o, endpoint=endpoint, dtype=dtype)
            assert_raises(ValueError, random.integers, low_o, high_o, endpoint=endpoint, dtype=dtype)