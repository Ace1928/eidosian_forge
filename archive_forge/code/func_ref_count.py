from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def ref_count(self):
    """
        Count the number of times each scope in this tree is referenced.

        Returns:
            dict[int, int]: Mapping of Scope instance ID to reference count
        """
    scope_ref_count = defaultdict(lambda: 0)
    for scope in self.traverse():
        for _, source in scope.selected_sources.values():
            scope_ref_count[id(source)] += 1
    return scope_ref_count