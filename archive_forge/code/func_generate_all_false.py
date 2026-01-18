import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def generate_all_false(self, dtype):
    arr = np.zeros((2, 2), [('junk', 'i1'), ('a', dtype)])
    arr.setflags(write=False)
    a = arr['a']
    assert_(not a.flags['C'])
    assert_(not a.flags['F'])
    assert_(not a.flags['O'])
    assert_(not a.flags['W'])
    assert_(not a.flags['A'])
    return a