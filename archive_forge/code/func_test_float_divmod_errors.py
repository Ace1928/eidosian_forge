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
@pytest.mark.xfail(sys.platform.startswith('darwin'), reason="MacOS seems to not give the correct 'invalid' warning for `fmod`.  Hopefully, others always do.")
@pytest.mark.parametrize('dtype', np.typecodes['Float'])
def test_float_divmod_errors(self, dtype):
    fzero = np.array(0.0, dtype=dtype)
    fone = np.array(1.0, dtype=dtype)
    finf = np.array(np.inf, dtype=dtype)
    fnan = np.array(np.nan, dtype=dtype)
    with np.errstate(divide='raise', invalid='ignore'):
        assert_raises(FloatingPointError, np.divmod, fone, fzero)
    with np.errstate(divide='ignore', invalid='raise'):
        assert_raises(FloatingPointError, np.divmod, fone, fzero)
    with np.errstate(invalid='raise'):
        assert_raises(FloatingPointError, np.divmod, fzero, fzero)
    with np.errstate(invalid='raise'):
        assert_raises(FloatingPointError, np.divmod, finf, finf)
    with np.errstate(divide='ignore', invalid='raise'):
        assert_raises(FloatingPointError, np.divmod, finf, fzero)
    with np.errstate(divide='raise', invalid='ignore'):
        np.divmod(finf, fzero)