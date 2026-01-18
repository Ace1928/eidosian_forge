from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload
def setter(self, fset: _SetterCallable[_T] | _SetterClassMethod[_T]) -> Self:
    self.fset = self._ensure_method(fset)
    return self