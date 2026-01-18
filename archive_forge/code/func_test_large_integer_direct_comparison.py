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
@pytest.mark.parametrize('dtypes', ['qQ', 'Qq'])
@pytest.mark.parametrize('py_comp, np_comp', [(operator.lt, np.less), (operator.le, np.less_equal), (operator.gt, np.greater), (operator.ge, np.greater_equal), (operator.eq, np.equal), (operator.ne, np.not_equal)])
@pytest.mark.parametrize('vals', [(2 ** 60, 2 ** 60 + 1), (2 ** 60 + 1, 2 ** 60)])
def test_large_integer_direct_comparison(self, dtypes, py_comp, np_comp, vals):
    a1 = np.array([2 ** 60], dtype=dtypes[0])
    a2 = np.array([2 ** 60 + 1], dtype=dtypes[1])
    expected = py_comp(2 ** 60, 2 ** 60 + 1)
    assert py_comp(a1, a2) == expected
    assert np_comp(a1, a2) == expected
    s1 = a1[0]
    s2 = a2[0]
    assert isinstance(s1, np.integer)
    assert isinstance(s2, np.integer)
    assert py_comp(s1, s2) == expected
    assert np_comp(s1, s2) == expected