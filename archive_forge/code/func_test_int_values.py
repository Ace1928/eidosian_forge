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
def test_int_values(self):
    exponents = [1, 2, 3, 5, 17, 0, -1, -2, -3]
    vals = [0, 1, 3, -1, -4, np.int8(-3), np.uint16(4)]
    self._check_pow(exponents, vals)