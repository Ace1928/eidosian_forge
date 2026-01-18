import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('fptosi')
def fptosi(self, value, typ, name=''):
    """
        Convert floating-point to signed integer:
            name = (typ) value
        """