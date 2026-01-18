from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
@property
def supported_table_args(self) -> t.Tuple[str, ...]:
    if not self._supported_table_args and self.mapping:
        depth = self.depth()
        if not depth:
            self._supported_table_args = tuple()
        elif 1 <= depth <= 3:
            self._supported_table_args = exp.TABLE_PARTS[:depth]
        else:
            raise SchemaError(f'Invalid mapping shape. Depth: {depth}')
    return self._supported_table_args