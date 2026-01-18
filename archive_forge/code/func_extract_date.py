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
def extract_date(cast: exp.Expression) -> t.Optional[t.Union[datetime.date, datetime.date]]:
    if isinstance(cast, exp.Cast):
        to = cast.to
    elif isinstance(cast, exp.TsOrDsToDate) and (not cast.args.get('format')):
        to = exp.DataType.build(exp.DataType.Type.DATE)
    else:
        return None
    if isinstance(cast.this, exp.Literal):
        value: t.Any = cast.this.name
    elif isinstance(cast.this, (exp.Cast, exp.TsOrDsToDate)):
        value = extract_date(cast.this)
    else:
        return None
    return cast_value(value, to)