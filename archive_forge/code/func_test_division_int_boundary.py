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
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('dtype,ex_val', itertools.product(np.sctypes['int'] + np.sctypes['uint'], ('np.array(range(fo.max-lsize, fo.max)).astype(dtype),np.arange(lsize).astype(dtype),range(15)', 'np.arange(fo.min, fo.min+lsize).astype(dtype),np.arange(lsize//-2, lsize//2).astype(dtype),range(fo.min, fo.min + 15)', 'np.array(range(fo.max-lsize, fo.max)).astype(dtype),np.arange(lsize).astype(dtype),[1,3,9,13,neg, fo.min+1, fo.min//2, fo.max//3, fo.max//4]')))
def test_division_int_boundary(self, dtype, ex_val):
    fo = np.iinfo(dtype)
    neg = -1 if fo.min < 0 else 1
    lsize = 512 + 7
    a, b, divisors = eval(ex_val)
    a_lst, b_lst = (a.tolist(), b.tolist())
    c_div = lambda n, d: 0 if d == 0 else fo.min if n and n == fo.min and (d == -1) else n // d
    with np.errstate(divide='ignore'):
        ac = a.copy()
        ac //= b
        div_ab = a // b
    div_lst = [c_div(x, y) for x, y in zip(a_lst, b_lst)]
    msg = 'Integer arrays floor division check (//)'
    assert all(div_ab == div_lst), msg
    msg_eq = 'Integer arrays floor division check (//=)'
    assert all(ac == div_lst), msg_eq
    for divisor in divisors:
        ac = a.copy()
        with np.errstate(divide='ignore', over='ignore'):
            div_a = a // divisor
            ac //= divisor
        div_lst = [c_div(i, divisor) for i in a_lst]
        assert all(div_a == div_lst), msg
        assert all(ac == div_lst), msg_eq
    with np.errstate(divide='raise', over='raise'):
        if 0 in b:
            with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
                a // b
        else:
            a // b
        if fo.min and fo.min in a:
            with pytest.raises(FloatingPointError, match='overflow encountered in floor_divide'):
                a // -1
        elif fo.min:
            a // -1
        with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
            a // 0
        with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
            ac = a.copy()
            ac //= 0
        np.array([], dtype=dtype) // 0