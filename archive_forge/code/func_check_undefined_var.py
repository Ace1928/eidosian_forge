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
def check_undefined_var(self, should_warn):

    @njit
    def foo(n):
        if n:
            if n > 0:
                c = 0
            return c
        else:
            c += 1
            return c
    if should_warn:
        with self.assertWarns(errors.NumbaWarning) as warns:
            self.check_func(foo, 1)
        self.assertIn('Detected uninitialized variable c', str(warns.warning))
    else:
        self.check_func(foo, 1)
    with self.assertRaises(UnboundLocalError):
        foo.py_func(0)