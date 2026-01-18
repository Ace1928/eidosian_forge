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
def set_and_check_flag(self, flag, dtype, arr):
    if dtype is None:
        dtype = arr.dtype
    b = np.require(arr, dtype, [flag])
    assert_(b.flags[flag])
    assert_(b.dtype == dtype)
    c = np.require(b, None, [flag])
    if flag[0] != 'O':
        assert_(c is b)
    else:
        assert_(c.flags[flag])