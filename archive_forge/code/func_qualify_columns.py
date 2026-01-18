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
def qualify_columns(expression: exp.Expression, schema: t.Dict | Schema, expand_alias_refs: bool=True, expand_stars: bool=True, infer_schema: t.Optional[bool]=None) -> exp.Expression:
    """
    Rewrite sqlglot AST to have fully qualified columns.

    Example:
        >>> import sqlglot
        >>> schema = {"tbl": {"col": "INT"}}
        >>> expression = sqlglot.parse_one("SELECT col FROM tbl")
        >>> qualify_columns(expression, schema).sql()
        'SELECT tbl.col AS col FROM tbl'

    Args:
        expression: Expression to qualify.
        schema: Database schema.
        expand_alias_refs: Whether to expand references to aliases.
        expand_stars: Whether to expand star queries. This is a necessary step
            for most of the optimizer's rules to work; do not set to False unless you
            know what you're doing!
        infer_schema: Whether to infer the schema if missing.

    Returns:
        The qualified expression.

    Notes:
        - Currently only handles a single PIVOT or UNPIVOT operator
    """
    schema = ensure_schema(schema)
    annotator = TypeAnnotator(schema)
    infer_schema = schema.empty if infer_schema is None else infer_schema
    dialect = Dialect.get_or_raise(schema.dialect)
    pseudocolumns = dialect.PSEUDOCOLUMNS
    for scope in traverse_scope(expression):
        resolver = Resolver(scope, schema, infer_schema=infer_schema)
        _pop_table_column_aliases(scope.ctes)
        _pop_table_column_aliases(scope.derived_tables)
        using_column_tables = _expand_using(scope, resolver)
        if schema.empty and expand_alias_refs:
            _expand_alias_refs(scope, resolver)
        _qualify_columns(scope, resolver)
        if not schema.empty and expand_alias_refs:
            _expand_alias_refs(scope, resolver)
        if not isinstance(scope.expression, exp.UDTF):
            if expand_stars:
                _expand_stars(scope, resolver, using_column_tables, pseudocolumns)
            qualify_outputs(scope)
        _expand_group_by(scope)
        _expand_order_by(scope, resolver)
        if dialect == 'bigquery':
            annotator.annotate_scope(scope)
    return expression