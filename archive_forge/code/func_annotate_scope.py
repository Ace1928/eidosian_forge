from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def annotate_scope(self, scope: Scope) -> None:
    selects = {}
    for name, source in scope.sources.items():
        if not isinstance(source, Scope):
            continue
        if isinstance(source.expression, exp.UDTF):
            values = []
            if isinstance(source.expression, exp.Lateral):
                if isinstance(source.expression.this, exp.Explode):
                    values = [source.expression.this.this]
            elif isinstance(source.expression, exp.Unnest):
                values = [source.expression]
            else:
                values = source.expression.expressions[0].expressions
            if not values:
                continue
            selects[name] = {alias: column for alias, column in zip(source.expression.alias_column_names, values)}
        else:
            selects[name] = {select.alias_or_name: select for select in source.expression.selects}
    for col in scope.columns:
        if not col.table:
            continue
        source = scope.sources.get(col.table)
        if isinstance(source, exp.Table):
            self._set_type(col, self.schema.get_column_type(source, col))
        elif source:
            if col.table in selects and col.name in selects[col.table]:
                self._set_type(col, selects[col.table][col.name].type)
            elif isinstance(source.expression, exp.Unnest):
                self._set_type(col, source.expression.type)
    self._maybe_annotate(scope.expression)