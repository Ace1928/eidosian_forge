from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_jitclass_instance_elements(self):
    JCItem = self.make_jitclass_element()

    def pyfunc(xs):
        xs[1], xs[0] = (xs[0], xs[1])
        return xs

    def eq(x, y):
        self.assertPreciseEqual(x.many, y.many)
        self.assertPreciseEqual(x.scalar, y.scalar)
    cfunc = jit(nopython=True)(pyfunc)
    arg = [JCItem(many=np.random.random(n + 1), scalar=n * 1.2) for n in range(5)]
    expect_arg = list(arg)
    got_arg = list(arg)
    expect_res = pyfunc(expect_arg)
    got_res = cfunc(got_arg)
    self.assert_list_element_with_tester(eq, expect_arg, got_arg)
    self.assert_list_element_with_tester(eq, expect_res, got_res)