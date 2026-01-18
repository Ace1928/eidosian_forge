from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@left_junction_char.setter
def left_junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('left_junction_char', val)
    self._left_junction_char = val