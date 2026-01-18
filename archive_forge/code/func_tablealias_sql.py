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
def tablealias_sql(self, e: exp.TableAlias) -> None:
    columns = e.columns
    if columns:
        self.stack.extend((')', columns, '('))
    self.stack.extend((e.this, ' AS '))