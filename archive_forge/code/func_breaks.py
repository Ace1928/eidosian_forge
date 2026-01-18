from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
@breaks.setter
def breaks(self, value: bool) -> None:
    self._options['breaks'] = value