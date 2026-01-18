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
def test_neq(self):

    def test_impl1():
        s = 'test'
        return s != 'test'

    def test_impl2():
        s = 'test1'
        return s != 'test'
    cfunc1 = jit(nopython=True)(test_impl1)
    cfunc2 = jit(nopython=True)(test_impl2)
    self.assertEqual(test_impl1(), cfunc1())
    self.assertEqual(test_impl2(), cfunc2())