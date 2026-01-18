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
@needs_blas
def check_matmul_npm(self, pyfunc):
    arrty = types.Array(types.float32, 1, 'C')
    cfunc = njit((arrty, arrty))(pyfunc)
    a = np.float32([1, 2])
    b = np.float32([3, 4])
    got = cfunc(a, b)
    self.assertPreciseEqual(got, np.dot(a, b))
    self.assertIsNot(got, a)
    self.assertIsNot(got, b)