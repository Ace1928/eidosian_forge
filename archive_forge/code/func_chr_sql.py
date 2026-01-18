from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def chr_sql(self, expression: exp.Chr) -> str:
    this = self.expressions(sqls=[expression.this] + expression.expressions)
    charset = expression.args.get('charset')
    using = f' USING {self.sql(charset)}' if charset else ''
    return f'CHAR({this}{using})'