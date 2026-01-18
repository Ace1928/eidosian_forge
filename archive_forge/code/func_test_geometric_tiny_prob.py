from numpy.testing import (assert_, assert_array_equal)
import numpy as np
import pytest
from numpy.random import Generator, MT19937
def test_geometric_tiny_prob(self):
    assert_array_equal(self.mt19937.geometric(p=1e-30, size=3), np.iinfo(np.int64).max)