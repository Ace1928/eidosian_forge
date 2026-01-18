from _weakref import (
from _weakrefset import WeakSet, _IterationGuard
import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
def keyrefs(self):
    """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
    return list(self.data)