import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_rank1_noteq(self):
    """Test two different array of rank 1 are found not equal."""
    a = np.array([1, 2])
    b = np.array([2, 2])
    self._test_not_equal(a, b)