from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def as_escaped_string(thing: Any, escape_chars: EscapeChars | None=escape_chars) -> str:
    translate = {'\t': '\\t', '\n': '\\n', '\r': '\\r'}
    if isinstance(escape_chars, dict):
        if default_escapes:
            translate.update(escape_chars)
        else:
            translate = escape_chars
        escape_chars = list(escape_chars.keys())
    else:
        escape_chars = escape_chars or ()
    result = str(thing)
    for c in escape_chars:
        result = result.replace(c, translate[c])
    return result