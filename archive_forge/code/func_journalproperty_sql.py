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
def journalproperty_sql(self, expression: exp.JournalProperty) -> str:
    no = 'NO ' if expression.args.get('no') else ''
    local = expression.args.get('local')
    local = f'{local} ' if local else ''
    dual = 'DUAL ' if expression.args.get('dual') else ''
    before = 'BEFORE ' if expression.args.get('before') else ''
    after = 'AFTER ' if expression.args.get('after') else ''
    return f'{no}{local}{dual}{before}{after}JOURNAL'