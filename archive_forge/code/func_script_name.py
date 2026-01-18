from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def script_name(code, default=KeyError):
    """Return the long, human-readable script name given a four-letter
    Unicode script code.

    If no matching name is found, a KeyError is raised by default.

    You can use the 'default' argument to return a fallback value (e.g.
    'Unknown' or None) instead of throwing an error.
    """
    try:
        return str(Scripts.NAMES[code].replace('_', ' '))
    except KeyError:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise
        return default