import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
@pytest.mark.parametrize('use_initial', [True, False])
def test_addition_reduce_negative_zero(dtype, use_initial):
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        neg_zero = dtype.type(complex(-0.0, -0.0))
    else:
        neg_zero = dtype.type(-0.0)
    kwargs = {}
    if use_initial:
        kwargs['initial'] = neg_zero
    else:
        pytest.xfail('-0. propagation in sum currently requires initial')
    for i in range(0, 150):
        arr = np.array([neg_zero] * i, dtype=dtype)
        res = np.sum(arr, **kwargs)
        if i > 0 or use_initial:
            assert _check_neg_zero(res)
        else:
            assert not np.signbit(res.real)
            assert not np.signbit(res.imag)