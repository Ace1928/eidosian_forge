import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_integers_max_int(self):
    actual = random.integers(np.iinfo('l').max, np.iinfo('l').max, endpoint=True)
    desired = np.iinfo('l').max
    assert_equal(actual, desired)