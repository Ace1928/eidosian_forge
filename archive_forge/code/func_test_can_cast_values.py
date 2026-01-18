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
def test_can_cast_values(self):
    for dt in np.sctypes['int'] + np.sctypes['uint']:
        ii = np.iinfo(dt)
        assert_(np.can_cast(ii.min, dt))
        assert_(np.can_cast(ii.max, dt))
        assert_(not np.can_cast(ii.min - 1, dt))
        assert_(not np.can_cast(ii.max + 1, dt))
    for dt in np.sctypes['float']:
        fi = np.finfo(dt)
        assert_(np.can_cast(fi.min, dt))
        assert_(np.can_cast(fi.max, dt))