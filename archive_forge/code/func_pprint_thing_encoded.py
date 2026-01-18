from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def pprint_thing_encoded(object, encoding: str='utf-8', errors: str='replace') -> bytes:
    value = pprint_thing(object)
    return value.encode(encoding, errors)