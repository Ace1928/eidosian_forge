from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@hrules.setter
def hrules(self, val) -> None:
    self._validate_option('hrules', val)
    self._hrules = val