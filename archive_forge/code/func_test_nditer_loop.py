import platform
import numpy as np
from numba import types
import unittest
from numba import njit
from numba.core import config
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess(envvars=_skylake_env)
def test_nditer_loop(self):

    def do_sum(x):
        acc = 0
        for v in np.nditer(x):
            acc += v.item()
        return acc
    llvm_ir = self.gen_ir(do_sum, (types.float64[::1],), fastmath=True)
    self.assertIn('vector.body', llvm_ir)
    self.assertIn('llvm.loop.isvectorized', llvm_ir)