from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def clear_rows(self) -> None:
    """Delete all rows from the table but keep the current field names"""
    self._rows = []
    self._dividers = []