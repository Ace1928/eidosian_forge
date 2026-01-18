import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_assigned(self):
    """Return *True* if a symbol is assigned to."""
    return bool(self.__flags & DEF_LOCAL)