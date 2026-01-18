import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def check_global_func_lifetime(self, **jitargs):
    c_f = jit(**jitargs)(global_usecase1)
    self.assertPreciseEqual(c_f(1), 2)
    cfunc = self.get_impl(c_f)
    wr = weakref.ref(c_f)
    refs = [weakref.ref(obj) for obj in (c_f, cfunc.__self__)]
    obj = c_f = cfunc = None
    gc.collect()
    self.assertEqual([wr() for wr in refs], [None] * len(refs))