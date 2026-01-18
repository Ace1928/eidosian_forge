import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_unop('fneg')
def fneg(self, arg, name='', flags=()):
    """
        Floating-point negative:
            name = -arg
        """