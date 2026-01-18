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
def check_matmul_objmode(self, pyfunc, inplace):
    cfunc = jit((), **force_pyobj_flags)(pyfunc)
    a = DumbMatrix(3)
    b = DumbMatrix(4)
    got = cfunc(a, b)
    self.assertEqual(got.value, 12)
    if inplace:
        self.assertIs(got, a)
    else:
        self.assertIsNot(got, a)
        self.assertIsNot(got, b)