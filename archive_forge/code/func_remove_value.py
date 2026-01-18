from __future__ import annotations
import logging # isort:skip
from typing import Generic, TypeVar, cast
def remove_value(self, key: K, value: V) -> None:
    """

        """
    if key is None:
        raise ValueError('Key is None')
    existing = self._dict.get(key)
    if isinstance(existing, set):
        existing = cast(set[V], existing)
        existing.discard(value)
        if len(existing) == 0:
            del self._dict[key]
    elif existing == value:
        del self._dict[key]