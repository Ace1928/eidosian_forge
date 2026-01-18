import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_const_dict_noninline_controlflow(self):
    """
        Tests generating large constant dict when one of the
        inputs has the change suggested in the error message
        for inlined control flow.
        """

    def non_inline_func(a, flag):
        val = 1 if flag else 2
        d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': val, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
        return d['S']
    py_func = non_inline_func
    cfunc = njit()(non_inline_func)
    value = 'a_string'
    a = py_func(value, False)
    b = cfunc(value, False)
    self.assertEqual(a, b)