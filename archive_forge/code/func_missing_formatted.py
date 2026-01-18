from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def missing_formatted(self, branches: bool=False) -> str:
    """The missing line numbers, formatted nicely.

        Returns a string like "1-2, 5-11, 13-14".

        If `branches` is true, includes the missing branch arcs also.

        """
    if branches and self.has_arcs():
        arcs = self.missing_branch_arcs().items()
    else:
        arcs = None
    return format_lines(self.statements, self.missing, arcs=arcs)