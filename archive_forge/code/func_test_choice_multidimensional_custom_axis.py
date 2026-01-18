import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_choice_multidimensional_custom_axis(self):
    random = Generator(MT19937(self.seed))
    actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
    desired = np.array([[0], [2], [4], [6]])
    assert_array_equal(actual, desired)