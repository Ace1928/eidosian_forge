import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_everything_returns_views(self):
    a = np.arange(5)
    assert_(a is not a[()])
    assert_(a is not a[...])
    assert_(a is not a[:])