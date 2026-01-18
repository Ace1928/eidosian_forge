import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def check_inner_function_lifetime(self, **jitargs):
    """
        When a jitted function calls into another jitted function, check
        that everything is collected as desired.
        """

    def mult_10(a):
        return a * 10
    c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
    c_mult_10.disable_compile()

    def do_math(x):
        return c_mult_10(x + 4)
    c_do_math = jit('intp(intp)', **jitargs)(do_math)
    c_do_math.disable_compile()
    self.assertEqual(c_do_math(1), 50)
    wrs = [weakref.ref(obj) for obj in (mult_10, c_mult_10, do_math, c_do_math, self.get_impl(c_mult_10).__self__, self.get_impl(c_do_math).__self__)]
    obj = mult_10 = c_mult_10 = do_math = c_do_math = None
    gc.collect()
    self.assertEqual([w() for w in wrs], [None] * len(wrs))