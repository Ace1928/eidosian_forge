import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_declared_global(self):
    """Return *True* if the symbol is declared global
        with a global statement."""
    return bool(self.__scope == GLOBAL_EXPLICIT)