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
def is_correlated_subquery(self):
    """Determine if this scope is a correlated subquery"""
    return bool((self.is_subquery or (self.parent and isinstance(self.parent.expression, exp.Lateral))) and self.external_columns)