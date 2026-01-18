from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def arcs_executed(self) -> list[TArc]:
    """Returns a sorted list of the arcs actually executed in the code."""
    executed: Iterable[TArc]
    executed = self.data.arcs(self.filename) or []
    executed = self.file_reporter.translate_arcs(executed)
    return sorted(executed)