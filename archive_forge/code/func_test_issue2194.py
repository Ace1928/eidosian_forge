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
def test_issue2194(self):

    @njit
    def foo():
        V = np.empty(1)
        s = np.uint32(1)
        for i in range(s):
            V[i] = 1
        for i in range(s, 1):
            pass
    self.check_func(foo)