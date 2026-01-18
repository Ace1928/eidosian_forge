from __future__ import annotations
from collections.abc import Callable, MutableMapping
import dataclasses as dc
from typing import Any, Literal
import warnings
from markdown_it._compat import DATACLASS_KWARGS
def attrPush(self, attrData: tuple[str, str | int | float]) -> None:
    """Add `[ name, value ]` attribute to list. Init attrs if necessary."""
    name, value = attrData
    self.attrSet(name, value)