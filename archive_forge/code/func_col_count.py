from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
@property
def col_count(self) -> int:
    """Number of dataframe columns to be summarized."""
    return self.info.col_count