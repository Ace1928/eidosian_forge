from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def best_len(values: list[str]) -> int:
    if values:
        return max((adj.len(x) for x in values))
    else:
        return 0