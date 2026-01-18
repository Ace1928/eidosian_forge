from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
@classmethod
def from_mapping_schema(cls, mapping_schema: MappingSchema) -> MappingSchema:
    return MappingSchema(schema=mapping_schema.mapping, visible=mapping_schema.visible, dialect=mapping_schema.dialect, normalize=mapping_schema.normalize)