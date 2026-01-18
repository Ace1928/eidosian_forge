import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_kws_inline_controlflow(self):
    """
        Tests generating large kws when one of the inputs
        has inlined controlflow.
        """

    def inline_func(flag):
        return sum_jit_func(arg0=1, arg1=1, arg2=1, arg3=1, arg4=1, arg5=1, arg6=1, arg7=1, arg8=1, arg9=1, arg10=1, arg11=1, arg12=1, arg13=1, arg14=1, arg15=1 if flag else 2)
    with self.assertRaises(UnsupportedError) as raises:
        njit()(inline_func)(False)
    self.assertIn('You can resolve this issue by moving the control flow out', str(raises.exception))