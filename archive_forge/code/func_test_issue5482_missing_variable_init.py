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
def test_issue5482_missing_variable_init(self):

    @njit('(intp, intp, intp)')
    def foo(x, v, n):
        for i in range(n):
            if i == 0:
                if i == x:
                    pass
                else:
                    problematic = v
            elif i == x:
                pass
            else:
                problematic = problematic + v
        return problematic