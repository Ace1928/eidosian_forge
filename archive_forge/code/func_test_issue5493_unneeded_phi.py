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
def test_issue5493_unneeded_phi(self):
    data = (np.ones(2), np.ones(2))
    A = np.ones(1)
    B = np.ones((1, 1))

    def foo(m, n, data):
        if len(data) == 1:
            v0 = data[0]
        else:
            v0 = data[0]
            for _ in range(1, len(data)):
                v0 += A
        for t in range(1, m):
            for idx in range(n):
                t = B
                if idx == 0:
                    if idx == n - 1:
                        pass
                    else:
                        problematic = t
                elif idx == n - 1:
                    pass
                else:
                    problematic = problematic + t
        return problematic
    expect = foo(10, 10, data)
    res1 = njit(foo)(10, 10, data)
    res2 = jit(forceobj=True, looplift=False)(foo)(10, 10, data)
    np.testing.assert_array_equal(expect, res1)
    np.testing.assert_array_equal(expect, res2)