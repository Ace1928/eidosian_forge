from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def adjoin(self, space: int, *lists, **kwargs) -> str:
    return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)