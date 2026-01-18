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
@pytest.mark.parametrize('dtype', np.sctypes['uint'] + np.sctypes['int'] + np.sctypes['float'] + [np.bool_])
@pytest.mark.parametrize('py_comp,np_comp', [(operator.lt, np.less), (operator.le, np.less_equal), (operator.gt, np.greater), (operator.ge, np.greater_equal), (operator.eq, np.equal), (operator.ne, np.not_equal)])
def test_comparison_functions(self, dtype, py_comp, np_comp):
    if dtype == np.bool_:
        a = np.random.choice(a=[False, True], size=1000)
        b = np.random.choice(a=[False, True], size=1000)
        scalar = True
    else:
        a = np.random.randint(low=1, high=10, size=1000).astype(dtype)
        b = np.random.randint(low=1, high=10, size=1000).astype(dtype)
        scalar = 5
    np_scalar = np.dtype(dtype).type(scalar)
    a_lst = a.tolist()
    b_lst = b.tolist()
    comp_b = np_comp(a, b).view(np.uint8)
    comp_b_list = [int(py_comp(x, y)) for x, y in zip(a_lst, b_lst)]
    comp_s1 = np_comp(np_scalar, b).view(np.uint8)
    comp_s1_list = [int(py_comp(scalar, x)) for x in b_lst]
    comp_s2 = np_comp(a, np_scalar).view(np.uint8)
    comp_s2_list = [int(py_comp(x, scalar)) for x in a_lst]
    assert_(comp_b.tolist() == comp_b_list, f'Failed comparison ({py_comp.__name__})')
    assert_(comp_s1.tolist() == comp_s1_list, f'Failed comparison ({py_comp.__name__})')
    assert_(comp_s2.tolist() == comp_s2_list, f'Failed comparison ({py_comp.__name__})')