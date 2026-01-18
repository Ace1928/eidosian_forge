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
def run_test_scalar_compare(self, pyfunc, flags=force_pyobj_flags, ordered=True):
    ops = self.compare_scalar_operands
    types_list = self.compare_types
    if not ordered:
        types_list = types_list + self.compare_unordered_types
    for typ in types_list:
        cfunc = jit((typ, typ), **flags)(pyfunc)
        for x, y in itertools.product(ops, ops):
            x = self.coerce_operand(x, typ)
            y = self.coerce_operand(y, typ)
            expected = pyfunc(x, y)
            got = cfunc(x, y)
            self.assertIs(type(got), type(expected))
            self.assertEqual(got, expected, 'mismatch with %r (%r, %r)' % (typ, x, y))