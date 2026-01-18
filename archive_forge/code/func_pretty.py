from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def pretty(self):
    """Print the output of ``choices()`` in a way that's easier to read."""
    out = ['Parser choices:']
    for k, v in self.choices().items():
        out.append('\t- %s -> %r' % (k, v))
    out.append('stack size: %s' % len(self.parser_state.state_stack))
    return '\n'.join(out)