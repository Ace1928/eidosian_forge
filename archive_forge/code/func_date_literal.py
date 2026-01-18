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
def date_literal(date, target_type=None):
    if not target_type or not target_type.is_type(*exp.DataType.TEMPORAL_TYPES):
        target_type = exp.DataType.Type.DATETIME if isinstance(date, datetime.datetime) else exp.DataType.Type.DATE
    return exp.cast(exp.Literal.string(date), target_type)