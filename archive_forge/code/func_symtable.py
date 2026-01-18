import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def symtable(code, filename, compile_type):
    """ Return the toplevel *SymbolTable* for the source code.

    *filename* is the name of the file with the code
    and *compile_type* is the *compile()* mode argument.
    """
    top = _symtable.symtable(code, filename, compile_type)
    return _newSymbolTable(top, filename)