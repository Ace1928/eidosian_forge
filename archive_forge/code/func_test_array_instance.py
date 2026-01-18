import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_array_instance(self):
    assert_equal(np.sctype2char(np.array([1.0, 2.0])), 'd')