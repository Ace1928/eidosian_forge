import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_free(self):
    """Return *True* if a referenced symbol is
        not assigned to.
        """
    return bool(self.__scope == FREE)