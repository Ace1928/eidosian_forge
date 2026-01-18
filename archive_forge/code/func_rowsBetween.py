from __future__ import annotations
import sys
import typing as t
from sqlglot import expressions as exp
from sqlglot.dataframe.sql import functions as F
from sqlglot.helper import flatten
def rowsBetween(self, start: int, end: int) -> WindowSpec:
    window_spec = self.copy()
    spec = self._calc_start_end(start, end)
    spec['kind'] = 'ROWS'
    window_spec.expression.set('spec', exp.WindowSpec(**{**window_spec.expression.args.get('spec', exp.WindowSpec()).args, **spec}))
    return window_spec