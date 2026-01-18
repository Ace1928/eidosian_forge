from __future__ import annotations
from typing import (
def raw_items(self) -> Iterator[Tuple[str, str]]:
    """
        Return an iterator of all values as ``(name, value)`` pairs.

        """
    return iter(self._list)