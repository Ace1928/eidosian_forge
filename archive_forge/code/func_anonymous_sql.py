from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def anonymous_sql(self, e: exp.Anonymous) -> None:
    this = e.this
    if isinstance(this, str):
        name = this.upper()
    elif isinstance(this, exp.Identifier):
        name = this.this
        name = f'"{name}"' if this.quoted else name.upper()
    else:
        raise ValueError(f"Anonymous.this expects a str or an Identifier, got '{this.__class__.__name__}'.")
    self.stack.extend((')', e.expressions, '(', name))