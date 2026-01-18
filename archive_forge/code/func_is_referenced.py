import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_referenced(self):
    """Return *True* if the symbol is used in
        its block.
        """
    return bool(self.__flags & _symtable.USE)