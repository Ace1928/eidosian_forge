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
def maybe_comment(self, sql: str, expression: t.Optional[exp.Expression]=None, comments: t.Optional[t.List[str]]=None, separated: bool=False) -> str:
    comments = (expression and expression.comments if comments is None else comments) if self.comments else None
    if not comments or isinstance(expression, self.EXCLUDE_COMMENTS):
        return sql
    comments_sql = ' '.join((f'/*{self.pad_comment(comment)}*/' for comment in comments if comment))
    if not comments_sql:
        return sql
    comments_sql = self._replace_line_breaks(comments_sql)
    if separated or isinstance(expression, self.WITH_SEPARATED_COMMENTS):
        return f'{self.sep()}{comments_sql}{sql}' if not sql or sql[0].isspace() else f'{comments_sql}{self.sep()}{sql}'
    return f'{sql} {comments_sql}'