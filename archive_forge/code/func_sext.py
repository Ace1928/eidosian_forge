import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('sext')
def sext(self, value, typ, name=''):
    """
        Sign-extending integer upcast to a larger type:
            name = (typ) value
        """