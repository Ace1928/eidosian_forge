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
def between_sql(self, e: exp.Between) -> None:
    self.stack.extend((e.args.get('high'), ' AND ', e.args.get('low'), ' BETWEEN ', e.this))