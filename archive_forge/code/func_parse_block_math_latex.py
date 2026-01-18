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
def parse_block_math_latex(self, m: Match[str], state: Any) -> Tuple[str, str]:
    """Parse block latex math ."""
    text = m.group(1)
    return ('block_math', text)