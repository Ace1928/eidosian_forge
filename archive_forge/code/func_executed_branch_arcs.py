from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def executed_branch_arcs(self) -> dict[TLineNo, list[TLineNo]]:
    """Return arcs that were executed from branch lines.

        Returns {l1:[l2a,l2b,...], ...}

        """
    executed = self.arcs_executed()
    branch_lines = set(self._branch_lines())
    eba = collections.defaultdict(list)
    for l1, l2 in executed:
        if l1 in branch_lines:
            eba[l1].append(l2)
    return eba