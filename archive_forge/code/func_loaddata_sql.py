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
def loaddata_sql(self, expression: exp.LoadData) -> str:
    local = ' LOCAL' if expression.args.get('local') else ''
    inpath = f' INPATH {self.sql(expression, 'inpath')}'
    overwrite = ' OVERWRITE' if expression.args.get('overwrite') else ''
    this = f' INTO TABLE {self.sql(expression, 'this')}'
    partition = self.sql(expression, 'partition')
    partition = f' {partition}' if partition else ''
    input_format = self.sql(expression, 'input_format')
    input_format = f' INPUTFORMAT {input_format}' if input_format else ''
    serde = self.sql(expression, 'serde')
    serde = f' SERDE {serde}' if serde else ''
    return f'LOAD DATA{local}{inpath}{overwrite}{this}{partition}{input_format}{serde}'