from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
def code_unescape(self, text: str) -> str:
    """Unescape code."""
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    return text