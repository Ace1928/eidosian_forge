import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_integers_closed(self):
    random = Generator(MT19937(self.seed))
    actual = random.integers(-99, 99, size=(3, 2), endpoint=True)
    desired = np.array([[-80, -56], [41, 38], [-83, -15]])
    assert_array_equal(actual, desired)