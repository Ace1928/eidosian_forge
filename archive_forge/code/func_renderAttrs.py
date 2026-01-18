from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
@staticmethod
def renderAttrs(token: Token) -> str:
    """Render token attributes to string."""
    result = ''
    for key, value in token.attrItems():
        result += ' ' + escapeHtml(key) + '="' + escapeHtml(str(value)) + '"'
    return result