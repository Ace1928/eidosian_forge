import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
@needs_blas
def test_corrcoef_edge_cases(self):
    pyfunc = corrcoef
    self.cov_corrcoef_edge_cases(pyfunc, first_arg_name='x')
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)
    for x in (np.nan, -np.inf, 3.142, 0):
        params = {'x': x}
        _check(params)