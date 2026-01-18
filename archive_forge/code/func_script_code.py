from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def script_code(script_name, default=KeyError):
    """Returns the four-letter Unicode script code from its long name

    If no matching script code is found, a KeyError is raised by default.

    You can use the 'default' argument to return a fallback string (e.g.
    'Zzzz' or None) instead of throwing an error.
    """
    normalized_name = _normalize_property_name(script_name)
    try:
        return _SCRIPT_CODES[normalized_name]
    except KeyError:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise
        return default