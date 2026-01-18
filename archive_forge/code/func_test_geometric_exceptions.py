import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_geometric_exceptions(self):
    assert_raises(ValueError, random.geometric, 1.1)
    assert_raises(ValueError, random.geometric, [1.1] * 10)
    assert_raises(ValueError, random.geometric, -0.1)
    assert_raises(ValueError, random.geometric, [-0.1] * 10)
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        assert_raises(ValueError, random.geometric, np.nan)
        assert_raises(ValueError, random.geometric, [np.nan] * 10)