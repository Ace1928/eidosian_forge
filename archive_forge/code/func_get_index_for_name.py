from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def get_index_for_name(self, name: str) -> int:
    """
        Return the index of the given name.
        """
    if name in self:
        self._sort()
        return self._priority.index([x for x in self._priority if x.name == name][0])
    raise ValueError('No item named "{}" exists.'.format(name))