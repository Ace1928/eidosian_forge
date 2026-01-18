from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def add_header_line(self) -> None:
    header_line = self.SPACING.join([_put_str(header, col_width) for header, col_width in zip(self.headers, self.gross_column_widths)])
    self._lines.append(header_line)