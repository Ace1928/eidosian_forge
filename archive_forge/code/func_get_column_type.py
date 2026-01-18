from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
def get_column_type(self, table: exp.Table | str, column: exp.Column | str, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> exp.DataType:
    normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)
    normalized_column_name = self._normalize_name(column if isinstance(column, str) else column.this, dialect=dialect, normalize=normalize)
    table_schema = self.find(normalized_table, raise_on_missing=False)
    if table_schema:
        column_type = table_schema.get(normalized_column_name)
        if isinstance(column_type, exp.DataType):
            return column_type
        elif isinstance(column_type, str):
            return self._to_data_type(column_type, dialect=dialect)
    return exp.DataType.build('unknown')