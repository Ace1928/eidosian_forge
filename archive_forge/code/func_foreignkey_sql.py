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
def foreignkey_sql(self, expression: exp.ForeignKey) -> str:
    expressions = self.expressions(expression, flat=True)
    reference = self.sql(expression, 'reference')
    reference = f' {reference}' if reference else ''
    delete = self.sql(expression, 'delete')
    delete = f' ON DELETE {delete}' if delete else ''
    update = self.sql(expression, 'update')
    update = f' ON UPDATE {update}' if update else ''
    return f'FOREIGN KEY ({expressions}){reference}{delete}{update}'