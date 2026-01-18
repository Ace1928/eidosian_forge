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
def compile_and_test(self, pyfunc, *args):
    from copy import deepcopy
    expect_args = deepcopy(args)
    expect = pyfunc(*expect_args)
    njit_args = deepcopy(args)
    cfunc = jit(nopython=True)(pyfunc)
    got = cfunc(*njit_args)
    self.assert_list_element_precise_equal(expect=expect, got=got)
    self.assert_list_element_precise_equal(expect=expect_args, got=njit_args)