from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def finalize_pending(self) -> None:
    for key, flag in self._pending_flags:
        self.set(key, flag, recursive=False)
    self._pending_flags.clear()