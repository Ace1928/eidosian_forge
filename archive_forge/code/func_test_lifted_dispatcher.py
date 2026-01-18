import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_lifted_dispatcher(self):

    @jit(forceobj=True)
    def foo():
        object()
        c = 0
        for i in range(10):
            c += i
        return c
    with ev.install_recorder('numba:compile') as rec:
        foo()
    self.assertGreaterEqual(len(rec.buffer), 4)
    cres = foo.overloads[foo.signatures[0]]
    [ldisp] = cres.lifted
    lifted_cres = ldisp.overloads[ldisp.signatures[0]]
    self.assertIsInstance(lifted_cres.metadata['timers']['compiler_lock'], float)
    self.assertIsInstance(lifted_cres.metadata['timers']['llvm_lock'], float)