import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_floor_division_complex(self):
    x = np.array([0.9 + 1j, -0.1 + 1j, 0.9 + 0.5 * 1j, 0.9 + 2.0 * 1j], dtype=np.complex128)
    with pytest.raises(TypeError):
        x // 7
    with pytest.raises(TypeError):
        np.divmod(x, 7)
    with pytest.raises(TypeError):
        np.remainder(x, 7)