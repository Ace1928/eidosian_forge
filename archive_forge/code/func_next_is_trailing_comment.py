from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def next_is_trailing_comment(config, tokens):
    """
  Return true if there is a trailing comment in the token stream
  """
    if not tokens:
        return False
    if next_is_explicit_trailing_comment(config, tokens):
        return True
    if is_valid_trailing_comment(tokens[0]):
        return True
    if len(tokens) < 2:
        return False
    if tokens[0].type == lex.TokenType.WHITESPACE and is_valid_trailing_comment(tokens[1]):
        return True
    return False