from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@preserve_internal_border.setter
def preserve_internal_border(self, val) -> None:
    self._validate_option('preserve_internal_border', val)
    self._preserve_internal_border = val