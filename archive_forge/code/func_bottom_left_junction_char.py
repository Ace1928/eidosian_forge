from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@bottom_left_junction_char.setter
def bottom_left_junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('bottom_left_junction_char', val)
    self._bottom_left_junction_char = val