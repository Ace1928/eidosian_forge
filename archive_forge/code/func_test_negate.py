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
def test_negate(self):
    pyfunc = self.op.negate_usecase
    values = [1, 2, 3, 1.2, 3.4j, True, False]
    cfunc = jit((), **force_pyobj_flags)(pyfunc)
    for val in values:
        self.assertEqual(pyfunc(val), cfunc(val))