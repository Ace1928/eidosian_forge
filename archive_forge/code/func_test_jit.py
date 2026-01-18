import math
import numpy as np
from numba.tests.support import captured_stdout, override_config
from numba import njit, vectorize, guvectorize
import unittest
def test_jit(self):

    def foo(x):
        return x + math.sin(x)
    fastfoo = njit(fastmath=True)(foo)
    slowfoo = njit(foo)
    self.assertEqual(fastfoo(0.5), slowfoo(0.5))
    fastllvm = fastfoo.inspect_llvm(fastfoo.signatures[0])
    slowllvm = slowfoo.inspect_llvm(slowfoo.signatures[0])
    self.assertIn('fadd fast', fastllvm)
    self.assertIn('call fast', fastllvm)
    self.assertNotIn('fadd fast', slowllvm)
    self.assertNotIn('call fast', slowllvm)