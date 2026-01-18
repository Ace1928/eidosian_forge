from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
@property
def srcCharCode(self) -> tuple[int, ...]:
    warnings.warn('StateBase.srcCharCode is deprecated. Use StateBase.src instead.', DeprecationWarning, stacklevel=2)
    if self._srcCharCode is None:
        self._srcCharCode = tuple((ord(c) for c in self._src))
    return self._srcCharCode