from __future__ import annotations
import typing as T
from ...interpreterbase import (
def iter_self(self) -> T.Iterator[int]:
    return iter(self.range)