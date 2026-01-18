from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def attrSet(self, name: str, value: str | int | float) -> None:
    """Set `name` attribute to `value`. Override old value if exists."""
    self.attrs[name] = value