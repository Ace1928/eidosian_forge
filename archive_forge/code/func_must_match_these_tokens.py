import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def must_match_these_tokens(s, l, t):
    theseTokens = _flatten(t.as_list())
    if theseTokens != matchTokens:
        raise ParseException(s, l, f'Expected {matchTokens}, found{theseTokens}')