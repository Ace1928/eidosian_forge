import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('sitofp')
def sitofp(self, value, typ, name=''):
    """
        Convert signed integer to floating-point:
            name = (typ) value
        """