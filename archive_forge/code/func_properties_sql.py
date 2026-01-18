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
def properties_sql(self, expression: exp.Properties) -> str:
    root_properties = []
    with_properties = []
    for p in expression.expressions:
        p_loc = self.PROPERTIES_LOCATION[p.__class__]
        if p_loc == exp.Properties.Location.POST_WITH:
            with_properties.append(p)
        elif p_loc == exp.Properties.Location.POST_SCHEMA:
            root_properties.append(p)
    return self.root_properties(exp.Properties(expressions=root_properties)) + self.with_properties(exp.Properties(expressions=with_properties))