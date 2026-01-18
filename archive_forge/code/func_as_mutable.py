from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def as_mutable(self):
    """Convert to an ``InteractiveParser``."""
    p = copy(self)
    return InteractiveParser(p.parser, p.parser_state, p.lexer_thread)