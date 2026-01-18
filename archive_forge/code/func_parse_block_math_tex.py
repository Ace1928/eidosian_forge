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
def parse_block_math_tex(self, m: Match[str], state: Any) -> Tuple[str, str]:
    """Parse block text math."""
    text = m.group(0)[2:-2]
    return ('block_math', text)