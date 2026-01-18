import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_fancy_on_read_only(self):

    class SubClass(np.ndarray):
        pass
    a = np.arange(5)
    s = a.view(SubClass)
    s.flags.writeable = False
    s_fancy = s[[0, 1, 2]]
    assert_(s_fancy.flags.writeable)