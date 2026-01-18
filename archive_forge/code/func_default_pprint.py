from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def default_pprint(thing: Any, max_seq_items: int | None=None) -> str:
    return pprint_thing(thing, escape_chars=('\t', '\r', '\n'), quote_strings=True, max_seq_items=max_seq_items)