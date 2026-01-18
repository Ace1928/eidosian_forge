import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_int64_uint64_corner_case(self):
    dt = np.int64
    tgt = np.iinfo(np.int64).max
    lbnd = np.int64(np.iinfo(np.int64).max)
    ubnd = np.uint64(np.iinfo(np.int64).max + 1)
    actual = random.randint(lbnd, ubnd, dtype=dt)
    assert_equal(actual, tgt)