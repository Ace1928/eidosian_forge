import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('sdiv')
def sdiv(self, lhs, rhs, name=''):
    """
        Signed integer division:
            name = lhs / rhs
        """