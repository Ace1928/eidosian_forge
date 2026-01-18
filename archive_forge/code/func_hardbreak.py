from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
def hardbreak(self, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    return '<br />\n' if options.xhtmlOut else '<br>\n'