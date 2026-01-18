import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def extract_constant(code, symbol, default=-1):
    """Extract the constant value of 'symbol' from 'code'

    If the name 'symbol' is bound to a constant value by the Python code
    object 'code', return that value.  If 'symbol' is bound to an expression,
    return 'default'.  Otherwise, return 'None'.

    Return value is based on the first assignment to 'symbol'.  'symbol' must
    be a global, or at least a non-"fast" local in the code block.  That is,
    only 'STORE_NAME' and 'STORE_GLOBAL' opcodes are checked, and 'symbol'
    must be present in 'code.co_names'.
    """
    if symbol not in code.co_names:
        return None
    name_idx = list(code.co_names).index(symbol)
    STORE_NAME = 90
    STORE_GLOBAL = 97
    LOAD_CONST = 100
    const = default
    for byte_code in dis.Bytecode(code):
        op = byte_code.opcode
        arg = byte_code.arg
        if op == LOAD_CONST:
            const = code.co_consts[arg]
        elif arg == name_idx and (op == STORE_NAME or op == STORE_GLOBAL):
            return const
        else:
            const = default