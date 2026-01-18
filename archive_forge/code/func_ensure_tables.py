from __future__ import annotations
import typing as t
from sqlglot.dialects.dialect import DialectType
from sqlglot.helper import dict_depth
from sqlglot.schema import AbstractMappingSchema, normalize_name
def ensure_tables(d: t.Optional[t.Dict], dialect: DialectType=None) -> Tables:
    return Tables(_ensure_tables(d, dialect=dialect))