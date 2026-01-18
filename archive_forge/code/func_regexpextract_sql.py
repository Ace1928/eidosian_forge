from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def regexpextract_sql(self, expression: exp.RegexpExtract) -> str:
    group = expression.args.get('group')
    parameters = expression.args.get('parameters') or (group and exp.Literal.string('c'))
    occurrence = expression.args.get('occurrence') or (parameters and exp.Literal.number(1))
    position = expression.args.get('position') or (occurrence and exp.Literal.number(1))
    return self.func('REGEXP_SUBSTR', expression.this, expression.expression, position, occurrence, parameters, group)