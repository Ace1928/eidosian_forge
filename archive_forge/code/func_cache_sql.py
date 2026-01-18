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
def cache_sql(self, expression: exp.Cache) -> str:
    lazy = ' LAZY' if expression.args.get('lazy') else ''
    table = self.sql(expression, 'this')
    options = expression.args.get('options')
    options = f' OPTIONS({self.sql(options[0])} = {self.sql(options[1])})' if options else ''
    sql = self.sql(expression, 'expression')
    sql = f' AS{self.sep()}{sql}' if sql else ''
    sql = f'CACHE{lazy} TABLE {table}{options}{sql}'
    return self.prepend_ctes(expression, sql)