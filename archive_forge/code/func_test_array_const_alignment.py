import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_array_const_alignment(self):
    """
        Issue #1933: the array declaration in the LLVM IR must have
        the right alignment specified.
        """
    sig = (types.intp,)
    cfunc = jit(sig, nopython=True)(getitem6)
    ir = cfunc.inspect_llvm(sig)
    for line in ir.splitlines():
        if 'XXXX_array_contents_XXXX' in line:
            self.assertIn('constant [24 x i8]', line)
            self.assertIn(', align 4', line)
            break
    else:
        self.fail('could not find array declaration in LLVM IR')