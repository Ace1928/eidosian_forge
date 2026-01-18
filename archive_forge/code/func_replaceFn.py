from __future__ import annotations
import logging
import re
from ..token import Token
from .state_core import StateCore
def replaceFn(match: re.Match[str]) -> str:
    return SCOPED_ABBR[match.group(1).lower()]