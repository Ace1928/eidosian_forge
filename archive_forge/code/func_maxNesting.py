from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
@maxNesting.setter
def maxNesting(self, value: int) -> None:
    self._options['maxNesting'] = value