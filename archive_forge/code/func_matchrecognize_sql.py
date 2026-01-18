from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def matchrecognize_sql(self, expression: exp.MatchRecognize) -> str:
    partition = self.partition_by_sql(expression)
    order = self.sql(expression, 'order')
    measures = self.expressions(expression, key='measures')
    measures = self.seg(f'MEASURES{self.seg(measures)}') if measures else ''
    rows = self.sql(expression, 'rows')
    rows = self.seg(rows) if rows else ''
    after = self.sql(expression, 'after')
    after = self.seg(after) if after else ''
    pattern = self.sql(expression, 'pattern')
    pattern = self.seg(f'PATTERN ({pattern})') if pattern else ''
    definition_sqls = [f'{self.sql(definition, 'alias')} AS {self.sql(definition, 'this')}' for definition in expression.args.get('define', [])]
    definitions = self.expressions(sqls=definition_sqls)
    define = self.seg(f'DEFINE{self.seg(definitions)}') if definitions else ''
    body = ''.join((partition, order, measures, rows, after, pattern, define))
    alias = self.sql(expression, 'alias')
    alias = f' {alias}' if alias else ''
    return f'{self.seg('MATCH_RECOGNIZE')} {self.wrap(body)}{alias}'