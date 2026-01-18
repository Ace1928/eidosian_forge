from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def qualify_outputs(scope_or_expression: Scope | exp.Expression) -> None:
    """Ensure all output columns are aliased"""
    if isinstance(scope_or_expression, exp.Expression):
        scope = build_scope(scope_or_expression)
        if not isinstance(scope, Scope):
            return
    else:
        scope = scope_or_expression
    new_selections = []
    for i, (selection, aliased_column) in enumerate(itertools.zip_longest(scope.expression.selects, scope.outer_columns)):
        if selection is None:
            break
        if isinstance(selection, exp.Subquery):
            if not selection.output_name:
                selection.set('alias', exp.TableAlias(this=exp.to_identifier(f'_col_{i}')))
        elif not isinstance(selection, exp.Alias) and (not selection.is_star):
            selection = alias(selection, alias=selection.output_name or f'_col_{i}', copy=False)
        if aliased_column:
            selection.set('alias', exp.to_identifier(aliased_column))
        new_selections.append(selection)
    if isinstance(scope.expression, exp.Select):
        scope.expression.set('expressions', new_selections)