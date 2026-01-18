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
def test_issue3931(self):

    @njit
    def foo(arr):
        for i in range(1):
            arr = arr.reshape(3 * 2)
            arr = arr.reshape(3, 2)
        return arr
    np.testing.assert_allclose(foo(np.zeros((3, 2))), foo.py_func(np.zeros((3, 2))))