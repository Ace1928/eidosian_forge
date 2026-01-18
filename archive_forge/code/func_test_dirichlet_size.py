import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_dirichlet_size(self):
    p = np.array([51.72840233779265, 39.74494232180944])
    assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
    assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
    assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
    assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
    assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
    assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))
    assert_raises(TypeError, random.dirichlet, p, float(1))