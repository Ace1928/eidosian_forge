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
def locate_properties(self, properties: exp.Properties) -> t.DefaultDict:
    properties_locs = defaultdict(list)
    for p in properties.expressions:
        p_loc = self.PROPERTIES_LOCATION[p.__class__]
        if p_loc != exp.Properties.Location.UNSUPPORTED:
            properties_locs[p_loc].append(p)
        else:
            self.unsupported(f'Unsupported property {p.key}')
    return properties_locs