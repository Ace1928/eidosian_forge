from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
def pivot_column_names(aggregations: t.List[exp.Expression], dialect: DialectType) -> t.List[str]:
    names = []
    for agg in aggregations:
        if isinstance(agg, exp.Alias):
            names.append(agg.alias)
        else:
            "\n            This case corresponds to aggregations without aliases being used as suffixes\n            (e.g. col_avg(foo)). We need to unquote identifiers because they're going to\n            be quoted in the base parser's `_parse_pivot` method, due to `to_identifier`.\n            Otherwise, we'd end up with `col_avg(`foo`)` (notice the double quotes).\n            "
            agg_all_unquoted = agg.transform(lambda node: exp.Identifier(this=node.name, quoted=False) if isinstance(node, exp.Identifier) else node)
            names.append(agg_all_unquoted.sql(dialect=dialect, normalize_functions='lower'))
    return names