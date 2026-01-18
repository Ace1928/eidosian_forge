import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
def test_issue5623_equal_statements_in_same_bb(self):

    def foo(pred, stack):
        i = 0
        c = 1
        if pred is True:
            stack[i] = c
            i += 1
            stack[i] = c
            i += 1
    python = np.array([0, 666])
    foo(True, python)
    nb = np.array([0, 666])
    njit(foo)(True, nb)
    expect = np.array([1, 1])
    np.testing.assert_array_equal(python, expect)
    np.testing.assert_array_equal(nb, expect)