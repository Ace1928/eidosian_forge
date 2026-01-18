import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('fptoui')
def fptoui(self, value, typ, name=''):
    """
        Convert floating-point to unsigned integer:
            name = (typ) value
        """