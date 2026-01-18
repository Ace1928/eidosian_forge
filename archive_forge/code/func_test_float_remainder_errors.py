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
@pytest.mark.skipif(hasattr(np.__config__, 'blas_ssl2_info'), reason='gh-22982')
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.xfail(sys.platform.startswith('darwin'), reason="MacOS seems to not give the correct 'invalid' warning for `fmod`.  Hopefully, others always do.")
@pytest.mark.parametrize('dtype', np.typecodes['Float'])
@pytest.mark.parametrize('fn', [np.fmod, np.remainder])
def test_float_remainder_errors(self, dtype, fn):
    fzero = np.array(0.0, dtype=dtype)
    fone = np.array(1.0, dtype=dtype)
    finf = np.array(np.inf, dtype=dtype)
    fnan = np.array(np.nan, dtype=dtype)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError, match='invalid value'):
            fn(fone, fzero)
        fn(fnan, fzero)
        fn(fzero, fnan)
        fn(fone, fnan)
        fn(fnan, fone)