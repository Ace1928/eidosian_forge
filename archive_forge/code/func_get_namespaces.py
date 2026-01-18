import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_namespaces(self):
    """Return a list of namespaces bound to this name"""
    return self.__namespaces