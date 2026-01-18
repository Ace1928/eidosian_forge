import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def check_3d(self, pyfunc, cfunc, generate_starargs):

    def check(a, b, c, args):
        self.check_stack(pyfunc, cfunc, (a, b, c) + args)

    def check_all_axes(a, b, c):
        for args in generate_starargs():
            check(a, b, c, args)
    a, b, c, d, e = self._3d_arrays()
    check_all_axes(a, b, b)
    check_all_axes(a, b, c)
    check_all_axes(a.T, b.T, a.T)
    check_all_axes(a.T, b.T, c.T)
    check_all_axes(a.T, b.T, d.T)
    check_all_axes(d.T, e.T, d.T)
    check_all_axes(a, b.astype(np.float64), b)