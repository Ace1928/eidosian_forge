import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def has_children(self):
    """Return *True* if the block has nested namespaces.
        """
    return bool(self._table.children)