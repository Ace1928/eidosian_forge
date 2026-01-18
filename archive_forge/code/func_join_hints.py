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
def join_hints(self):
    """
        Hints that exist in the scope that reference tables

        Returns:
            list[exp.JoinHint]: Join hints that are referenced within the scope
        """
    if self._join_hints is None:
        return []
    return self._join_hints