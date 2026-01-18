import _symtable
from _symtable import (USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM,
import weakref
def get_identifiers(self):
    """Return a view object containing the names of symbols in the table.
        """
    return self._table.symbols.keys()