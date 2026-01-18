from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
from cmakelang import common
def get_first_non_whitespace_token(tokens):
    """
  Return the first token in the list that is not whitespace, or None
  """
    for token in tokens:
        if token.type not in (TokenType.WHITESPACE, TokenType.NEWLINE):
            return token
    return None