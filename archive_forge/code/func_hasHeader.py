from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def hasHeader(self, name: AnyStr) -> bool:
    """
        Check for the existence of a given header.

        @param name: The name of the HTTP header to check for.

        @return: C{True} if the header exists, otherwise C{False}.
        """
    return self._encodeName(name) in self._rawHeaders