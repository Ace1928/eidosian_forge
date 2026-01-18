from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def make_fields_unique(self, fields) -> None:
    """
        iterates over the row and make each field unique
        """
    for i in range(0, len(fields)):
        for j in range(i + 1, len(fields)):
            if fields[i] == fields[j]:
                fields[j] += "'"