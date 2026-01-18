from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
def flatten_schema(schema: t.Dict, depth: int, keys: t.Optional[t.List[str]]=None) -> t.List[t.List[str]]:
    tables = []
    keys = keys or []
    for k, v in schema.items():
        if depth >= 2:
            tables.extend(flatten_schema(v, depth - 1, keys + [k]))
        elif depth == 1:
            tables.append(keys + [k])
    return tables