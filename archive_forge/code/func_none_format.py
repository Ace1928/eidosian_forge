from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@none_format.setter
def none_format(self, val):
    if not self._field_names:
        self._none_format = {}
    elif val is None or (isinstance(val, dict) and len(val) == 0):
        for field in self._field_names:
            self._none_format[field] = None
    else:
        self._validate_none_format(val)
        for field in self._field_names:
            self._none_format[field] = val