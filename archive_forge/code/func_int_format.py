from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@int_format.setter
def int_format(self, val) -> None:
    if val is None or (isinstance(val, dict) and len(val) == 0):
        self._int_format = {}
    else:
        self._validate_option('int_format', val)
        for field in self._field_names:
            self._int_format[field] = val