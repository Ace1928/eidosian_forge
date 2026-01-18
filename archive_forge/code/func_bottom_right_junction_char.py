from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@bottom_right_junction_char.setter
def bottom_right_junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('bottom_right_junction_char', val)
    self._bottom_right_junction_char = val