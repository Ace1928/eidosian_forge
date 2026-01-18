from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@right_junction_char.setter
def right_junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('right_junction_char', val)
    self._right_junction_char = val