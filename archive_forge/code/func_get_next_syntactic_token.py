from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def get_next_syntactic_token(tokens):
    """
  return the first non-whitespace token in the list
  """
    for token in iter_syntactic_tokens(tokens):
        return token
    return None