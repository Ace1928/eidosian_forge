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
def insert_sql(self, expression: exp.Insert) -> str:
    hint = self.sql(expression, 'hint')
    overwrite = expression.args.get('overwrite')
    if isinstance(expression.this, exp.Directory):
        this = ' OVERWRITE' if overwrite else ' INTO'
    else:
        this = self.INSERT_OVERWRITE if overwrite else ' INTO'
    stored = self.sql(expression, 'stored')
    stored = f' {stored}' if stored else ''
    alternative = expression.args.get('alternative')
    alternative = f' OR {alternative}' if alternative else ''
    ignore = ' IGNORE' if expression.args.get('ignore') else ''
    is_function = expression.args.get('is_function')
    if is_function:
        this = f'{this} FUNCTION'
    this = f'{this} {self.sql(expression, 'this')}'
    exists = ' IF EXISTS' if expression.args.get('exists') else ''
    where = self.sql(expression, 'where')
    where = f'{self.sep()}REPLACE WHERE {where}' if where else ''
    expression_sql = f'{self.sep()}{self.sql(expression, 'expression')}'
    on_conflict = self.sql(expression, 'conflict')
    on_conflict = f' {on_conflict}' if on_conflict else ''
    by_name = ' BY NAME' if expression.args.get('by_name') else ''
    returning = self.sql(expression, 'returning')
    if self.RETURNING_END:
        expression_sql = f'{expression_sql}{on_conflict}{returning}'
    else:
        expression_sql = f'{returning}{expression_sql}{on_conflict}'
    sql = f'INSERT{hint}{alternative}{ignore}{this}{stored}{by_name}{exists}{where}{expression_sql}'
    return self.prepend_ctes(expression, sql)