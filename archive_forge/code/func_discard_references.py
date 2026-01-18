import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
def discard_references(self, expression, key):
    """Remove all references to parameters contained within ``expression`` at the given table
        ``key``.  This also discards parameter entries from the table if they have no further
        references.  No action is taken if the object is not tracked."""
    for parameter in expression.parameters:
        if (refs := self._table.get(parameter)) is not None:
            if len(refs) == 1:
                del self[parameter]
            else:
                refs.discard(key)