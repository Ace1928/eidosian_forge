from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def get_first_semantic_token(tokens):
    """
  Return the first token with semantic meaning
  """
    return get_nth_semantic_token(tokens, 0)