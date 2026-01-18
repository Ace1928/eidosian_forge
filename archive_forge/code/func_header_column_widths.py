from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
@property
def header_column_widths(self) -> Sequence[int]:
    """Widths of header columns (only titles)."""
    return [len(col) for col in self.headers]