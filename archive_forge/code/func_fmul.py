import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('fmul')
def fmul(self, lhs, rhs, name=''):
    """
        Floating-point multiplication:
            name = lhs * rhs
        """