import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def is_parameter(self):
    """Return *True* if the symbol is a parameter.
        """
    return bool(self.__flags & DEF_PARAM)