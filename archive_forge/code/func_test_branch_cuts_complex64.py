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
@pytest.mark.xfail(IS_MUSL, reason='gh23049')
@pytest.mark.xfail(IS_WASM, reason="doesn't work")
def test_branch_cuts_complex64(self):
    _check_branch_cut(np.log, -0.5, 1j, 1, -1, True, np.complex64)
    _check_branch_cut(np.log2, -0.5, 1j, 1, -1, True, np.complex64)
    _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True, np.complex64)
    _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True, np.complex64)
    _check_branch_cut(np.sqrt, -0.5, 1j, 1, -1, True, np.complex64)
    _check_branch_cut(np.arcsin, [-2, 2], [1j, 1j], 1, -1, True, np.complex64)
    _check_branch_cut(np.arccos, [-2, 2], [1j, 1j], 1, -1, True, np.complex64)
    _check_branch_cut(np.arctan, [0 - 2j, 2j], [1, 1], -1, 1, True, np.complex64)
    _check_branch_cut(np.arcsinh, [0 - 2j, 2j], [1, 1], -1, 1, True, np.complex64)
    _check_branch_cut(np.arccosh, [-1, 0.5], [1j, 1j], 1, -1, True, np.complex64)
    _check_branch_cut(np.arctanh, [-2, 2], [1j, 1j], 1, -1, True, np.complex64)
    _check_branch_cut(np.arcsin, [0 - 2j, 2j], [1, 1], 1, 1, False, np.complex64)
    _check_branch_cut(np.arccos, [0 - 2j, 2j], [1, 1], 1, 1, False, np.complex64)
    _check_branch_cut(np.arctan, [-2, 2], [1j, 1j], 1, 1, False, np.complex64)
    _check_branch_cut(np.arcsinh, [-2, 2, 0], [1j, 1j, 1], 1, 1, False, np.complex64)
    _check_branch_cut(np.arccosh, [0 - 2j, 2j, 2], [1, 1, 1j], 1, 1, False, np.complex64)
    _check_branch_cut(np.arctanh, [0 - 2j, 2j, 0], [1, 1, 1j], 1, 1, False, np.complex64)