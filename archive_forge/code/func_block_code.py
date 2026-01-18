import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
def block_code(self, code: str, info: Optional[str]=None) -> str:
    """Handle block code."""
    lang: Optional[str] = ''
    lexer: Optional[Lexer] = None
    if info:
        if info.startswith('mermaid'):
            return self.block_mermaidjs(code)
        try:
            if info.strip().split(None, 1):
                lang = info.strip().split(maxsplit=1)[0]
                lexer = get_lexer_by_name(lang, stripall=True)
        except ClassNotFound:
            code = f'{lang}\n{code}'
            lang = None
    if not lang:
        return super().block_code(code, info=info)
    formatter = HtmlFormatter()
    return highlight(code, lexer, formatter)