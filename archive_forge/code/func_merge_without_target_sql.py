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
def merge_without_target_sql(self: Generator, expression: exp.Merge) -> str:
    """Remove table refs from columns in when statements."""
    alias = expression.this.args.get('alias')

    def normalize(identifier: t.Optional[exp.Identifier]) -> t.Optional[str]:
        return self.dialect.normalize_identifier(identifier).name if identifier else None
    targets = {normalize(expression.this.this)}
    if alias:
        targets.add(normalize(alias.this))
    for when in expression.expressions:
        when.transform(lambda node: exp.column(node.this) if isinstance(node, exp.Column) and normalize(node.args.get('table')) in targets else node, copy=False)
    return self.merge_sql(expression)