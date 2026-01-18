import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('fpext')
def fpext(self, value, typ, name=''):
    """
        Floating-point upcast to a more precise type:
            name = (typ) value
        """