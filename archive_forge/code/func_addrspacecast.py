import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('addrspacecast')
def addrspacecast(self, value, typ, name=''):
    """
        Pointer cast to a different address space:
            name = (typ) value
        """