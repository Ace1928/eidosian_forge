import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_invalid_array_shape(self):
    assert_raises(ValueError, random.RandomState, np.array([], dtype=np.int64))
    assert_raises(ValueError, random.RandomState, [[1, 2, 3]])
    assert_raises(ValueError, random.RandomState, [[1, 2, 3], [4, 5, 6]])