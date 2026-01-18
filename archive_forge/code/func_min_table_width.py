from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@min_table_width.setter
def min_table_width(self, val) -> None:
    self._validate_option('min_table_width', val)
    self._min_table_width = val