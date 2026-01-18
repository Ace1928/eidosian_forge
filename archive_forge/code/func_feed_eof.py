from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def feed_eof(self, last_token=None):
    """Feed a '$END' Token. Borrows from 'last_token' if given."""
    eof = Token.new_borrow_pos('$END', '', last_token) if last_token is not None else self.lexer_thread._Token('$END', '', 0, 1, 1)
    return self.feed_token(eof)