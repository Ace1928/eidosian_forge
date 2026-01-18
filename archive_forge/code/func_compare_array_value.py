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
def compare_array_value(self, dz, value, fill_value):
    if value is not None:
        if fill_value:
            z = np.array(value).astype(dz.dtype)
            assert_(np.all(dz == z))
        else:
            assert_(np.all(dz == value))