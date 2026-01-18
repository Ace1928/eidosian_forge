from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@horizontal_align_char.setter
def horizontal_align_char(self, val) -> None:
    val = str(val)
    self._validate_option('horizontal_align_char', val)
    self._horizontal_align_char = val