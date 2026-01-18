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
def external_columns(self):
    """
        Columns that appear to reference sources in outer scopes.

        Returns:
            list[exp.Column]: Column instances that don't reference
                sources in the current scope.
        """
    if self._external_columns is None:
        if isinstance(self.expression, exp.Union):
            left, right = self.union_scopes
            self._external_columns = left.external_columns + right.external_columns
        else:
            self._external_columns = [c for c in self.columns if c.table not in self.selected_sources]
    return self._external_columns