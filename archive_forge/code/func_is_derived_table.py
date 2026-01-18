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
def is_derived_table(self):
    """Determine if this scope is a derived table"""
    return self.scope_type == ScopeType.DERIVED_TABLE