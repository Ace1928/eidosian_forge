import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('frem')
def frem(self, lhs, rhs, name=''):
    """
        Floating-point remainder:
            name = lhs % rhs
        """