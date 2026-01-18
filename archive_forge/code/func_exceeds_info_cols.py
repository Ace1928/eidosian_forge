from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
@property
def exceeds_info_cols(self) -> bool:
    """Check if number of columns to be summarized does not exceed maximum."""
    return bool(self.col_count > self.max_cols)