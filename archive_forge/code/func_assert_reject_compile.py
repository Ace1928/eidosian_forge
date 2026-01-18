import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def assert_reject_compile(pyfunc, argtypes, opname):
    msg = 'expecting TypingError when compiling {}'.format(pyfunc)
    with self.assertRaises(errors.TypingError, msg=msg) as raises:
        njit(argtypes)(pyfunc)
    fmt = _header_lead + ' {}'
    expecting = fmt.format(opname if isinstance(opname, str) else 'Function({})'.format(opname))
    self.assertIn(expecting, str(raises.exception))