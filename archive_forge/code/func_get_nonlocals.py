import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_nonlocals(self):
    """Return a tuple of nonlocals in the function.
        """
    if self.__nonlocals is None:
        self.__nonlocals = self.__idents_matching(lambda x: x & DEF_NONLOCAL)
    return self.__nonlocals