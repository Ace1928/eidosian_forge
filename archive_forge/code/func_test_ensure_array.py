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
def test_ensure_array(self):

    class ArraySubclass(np.ndarray):
        pass
    a = ArraySubclass((2, 2))
    b = np.require(a, None, ['E'])
    assert_(type(b) is np.ndarray)