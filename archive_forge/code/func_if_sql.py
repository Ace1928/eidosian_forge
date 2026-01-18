from __future__ import annotations
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import Dialect, rename_func
def if_sql(self, expression: exp.If) -> str:
    this = self.sql(expression, 'this')
    true = self.sql(expression, 'true')
    false = self.sql(expression, 'false')
    return f'IF {this} THEN {true} ELSE {false} END'