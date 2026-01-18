import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop_with_overflow('sadd')
def sadd_with_overflow(self, lhs, rhs, name=''):
    """
        Signed integer addition with overflow:
            name = {result, overflow bit} = lhs + rhs
        """