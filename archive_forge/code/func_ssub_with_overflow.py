import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop_with_overflow('ssub')
def ssub_with_overflow(self, lhs, rhs, name=''):
    """
        Signed integer subtraction with overflow:
            name = {result, overflow bit} = lhs - rhs
        """