from __future__ import annotations
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.spark import Spark
from sqlglot.tokens import TokenType
def generatedasidentitycolumnconstraint_sql(self, expression: exp.GeneratedAsIdentityColumnConstraint) -> str:
    expression.set('this', True)
    return super().generatedasidentitycolumnconstraint_sql(expression)