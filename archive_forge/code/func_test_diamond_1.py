import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
@TestCase.run_test_in_subprocess
def test_diamond_1(self):

    def func(n):
        a = np.ones(n)
        x = 0
        if n > 2:
            x = a.sum()
        return x + 1
    with set_refprune_flags('per_bb,diamond'):
        self.check(func, types.intp, basicblock=True, diamond=True, fanout=False, fanout_raise=False)