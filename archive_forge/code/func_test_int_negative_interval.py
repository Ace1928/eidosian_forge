import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_int_negative_interval(self):
    assert_(-5 <= random.randint(-5, -1) < -1)
    x = random.randint(-5, -1, 5)
    assert_(np.all(-5 <= x))
    assert_(np.all(x < -1))