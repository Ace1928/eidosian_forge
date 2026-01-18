from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def indexcolumnconstraint_sql(self, expression: exp.IndexColumnConstraint) -> str:
    this = self.sql(expression, 'this')
    this = f' {this}' if this else ''
    expr = self.sql(expression, 'expression')
    expr = f' {expr}' if expr else ''
    index_type = self.sql(expression, 'index_type')
    index_type = f' TYPE {index_type}' if index_type else ''
    granularity = self.sql(expression, 'granularity')
    granularity = f' GRANULARITY {granularity}' if granularity else ''
    return f'INDEX{this}{expr}{index_type}{granularity}'