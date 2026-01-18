from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def altertable_sql(self, expression: exp.AlterTable) -> str:
    actions = expression.args['actions']
    if isinstance(actions[0], exp.RenameTable):
        table = self.sql(expression.this)
        target = actions[0].this
        target = self.sql(exp.table_(target.this) if isinstance(target, exp.Table) else target)
        return f"EXEC sp_rename '{table}', '{target}'"
    return super().altertable_sql(expression)