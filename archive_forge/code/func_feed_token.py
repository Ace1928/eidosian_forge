from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
def feed_token(self, token):
    c = copy(self)
    c.result = InteractiveParser.feed_token(c, token)
    return c