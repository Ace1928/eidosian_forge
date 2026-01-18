from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce
from typing import Generic, TypeVar
def append_plist(self, pl):
    return self._append(pl, lambda l: l)