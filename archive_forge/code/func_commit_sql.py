from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def commit_sql(self, expression: exp.Commit) -> str:
    this = self.sql(expression, 'this')
    this = f' {this}' if this else ''
    durability = expression.args.get('durability')
    durability = f' WITH (DELAYED_DURABILITY = {('ON' if durability else 'OFF')})' if durability is not None else ''
    return f'COMMIT TRANSACTION{this}{durability}'