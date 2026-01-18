import numpy as np
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type
def gdb_init(*args):
    """
    Calling this function will invoke gdb and attach it to the current process
    at the call site, then continue executing the process under gdb's control.
    Arguments are strings in the gdb command language syntax which will be
    executed by gdb once initialisation has occurred.
    """
    _gdb_python_call_gen('gdb_init', *args)()