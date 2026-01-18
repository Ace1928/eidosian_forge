from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re
def substitute_match(m: re.Match[str]) -> str:
    key = m.group(0)
    if key not in replacements:
        if key[3:-4] in replacements:
            return f'<p>{replacements[key[3:-4]]}</p>'
        else:
            return key
    return replacements[key]