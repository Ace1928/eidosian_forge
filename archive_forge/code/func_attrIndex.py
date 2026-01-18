from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def attrIndex(self, name: str) -> int:
    warnings.warn('Token.attrIndex should not be used, since Token.attrs is a dictionary', UserWarning)
    if name not in self.attrs:
        return -1
    return list(self.attrs.keys()).index(name)