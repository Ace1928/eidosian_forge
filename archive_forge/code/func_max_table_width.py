from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@max_table_width.setter
def max_table_width(self, val) -> None:
    self._validate_option('max_table_width', val)
    self._max_table_width = val