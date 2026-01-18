from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def arc_possibilities(self) -> list[TArc]:
    """Returns a sorted list of the arcs in the code."""
    return self._arc_possibilities