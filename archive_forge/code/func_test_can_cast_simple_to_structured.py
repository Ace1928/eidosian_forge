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
def test_can_cast_simple_to_structured(self):
    assert_(not np.can_cast('i4', 'i4,i4'))
    assert_(not np.can_cast('i4', 'i4,i2'))
    assert_(np.can_cast('i4', 'i4,i4', casting='unsafe'))
    assert_(np.can_cast('i4', 'i4,i2', casting='unsafe'))
    assert_(not np.can_cast('i2', [('f1', 'i4')]))
    assert_(not np.can_cast('i2', [('f1', 'i4')], casting='same_kind'))
    assert_(np.can_cast('i2', [('f1', 'i4')], casting='unsafe'))
    assert_(not np.can_cast('i2', [('f1', 'i4,i4')]))
    assert_(np.can_cast('i2', [('f1', 'i4,i4')], casting='unsafe'))
    assert_(not np.can_cast('i2', [('f1', '(2,3)i4')]))
    assert_(np.can_cast('i2', [('f1', '(2,3)i4')], casting='unsafe'))