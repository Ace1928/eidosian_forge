import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_choice_uniform_noreplace(self):
    random.seed(self.seed)
    actual = random.choice(4, 3, replace=False)
    desired = np.array([0, 1, 3])
    assert_array_equal(actual, desired)