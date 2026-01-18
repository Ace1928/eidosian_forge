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
def test_get_slice(self):

    def pyfunc():
        con = []
        for i in range(5):
            con.append(np.arange(2))
        return con[2:4]
    cfunc = jit(nopython=True)(pyfunc)
    expect = pyfunc()
    got = cfunc()
    self.assert_list_element_precise_equal(expect=expect, got=got)