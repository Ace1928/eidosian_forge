from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def iter_syntactic_tokens(tokens):
    """
  Return a generator over the list of tokens yielding only those that are
  not whitespace
  """
    skip_tokens = (lex.TokenType.WHITESPACE, lex.TokenType.NEWLINE)
    for token in tokens:
        if token.type in skip_tokens:
            continue
        yield token