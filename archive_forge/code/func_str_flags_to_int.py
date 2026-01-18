from __future__ import annotations
import re
from typing import Any, Generic, Pattern, Type, TypeVar, Union
from bson._helpers import _getstate_slots, _setstate_slots
from bson.son import RE_TYPE
def str_flags_to_int(str_flags: str) -> int:
    flags = 0
    if 'i' in str_flags:
        flags |= re.IGNORECASE
    if 'l' in str_flags:
        flags |= re.LOCALE
    if 'm' in str_flags:
        flags |= re.MULTILINE
    if 's' in str_flags:
        flags |= re.DOTALL
    if 'u' in str_flags:
        flags |= re.UNICODE
    if 'x' in str_flags:
        flags |= re.VERBOSE
    return flags