from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def rowformatserdeproperty_sql(self, expression: exp.RowFormatSerdeProperty) -> str:
    serde_props = self.sql(expression, 'serde_properties')
    serde_props = f' {serde_props}' if serde_props else ''
    return f'ROW FORMAT SERDE {self.sql(expression, 'this')}{serde_props}'