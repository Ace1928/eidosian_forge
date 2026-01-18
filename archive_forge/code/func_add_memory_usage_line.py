from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def add_memory_usage_line(self) -> None:
    """Add line containing memory usage."""
    self._lines.append(f'memory usage: {self.memory_usage_string}')