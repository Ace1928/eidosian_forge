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
def latex_environment(self, name: str, body: str) -> str:
    """Handle a latex environment."""
    name, body = (self.escape_html(name), self.escape_html(body))
    return f'\\begin{{{name}}}{body}\\end{{{name}}}'