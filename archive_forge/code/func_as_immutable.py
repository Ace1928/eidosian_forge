from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def as_immutable(self):
    """Convert to an ``ImmutableInteractiveParser``."""
    p = copy(self)
    return ImmutableInteractiveParser(p.parser, p.parser_state, p.lexer_thread)