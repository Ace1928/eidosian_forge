from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.helper import tsort
def other_table_names(join: exp.Join) -> t.Set[str]:
    on = join.args.get('on')
    return exp.column_table_names(on, join.alias_or_name) if on else set()