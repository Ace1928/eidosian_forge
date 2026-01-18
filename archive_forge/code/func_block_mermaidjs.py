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
def block_mermaidjs(self, code: str) -> str:
    """Handle mermaid syntax."""
    return f'<div class="jp-Mermaid"><pre class="mermaid">\n{code.strip()}\n</pre></div>'