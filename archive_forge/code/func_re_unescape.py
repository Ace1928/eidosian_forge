import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def re_unescape(s: str) -> str:
    """Unescape a string escaped by `re.escape`.

    May raise ``ValueError`` for regular expressions which could not
    have been produced by `re.escape` (for example, strings containing
    ``\\d`` cannot be unescaped).

    .. versionadded:: 4.4
    """
    return _re_unescape_pattern.sub(_re_unescape_replacement, s)