from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def isMdAsciiPunct(ch: int) -> bool:
    """Markdown ASCII punctuation characters.

    ::

        !, ", #, $, %, &, ', (, ), *, +, ,, -, ., /, :, ;, <, =, >, ?, @, [, \\, ], ^, _, `, {, |, }, or ~

    See http://spec.commonmark.org/0.15/#ascii-punctuation-character

    Don't confuse with unicode punctuation !!! It lacks some chars in ascii range.

    """
    return ch in MD_ASCII_PUNCT