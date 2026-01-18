import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('fsub')
def fsub(self, lhs, rhs, name=''):
    """
        Floating-point subtraction:
            name = lhs - rhs
        """