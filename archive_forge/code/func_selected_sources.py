from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
@property
def selected_sources(self):
    """
        Mapping of nodes and sources that are actually selected from in this scope.

        That is, all tables in a schema are selectable at any point. But a
        table only becomes a selected source if it's included in a FROM or JOIN clause.

        Returns:
            dict[str, (exp.Table|exp.Select, exp.Table|Scope)]: selected sources and nodes
        """
    if self._selected_sources is None:
        result = {}
        for name, node in self.references:
            if name in result:
                raise OptimizeError(f'Alias already used: {name}')
            if name in self.sources:
                result[name] = (node, self.sources[name])
        self._selected_sources = result
    return self._selected_sources