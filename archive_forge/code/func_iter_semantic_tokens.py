from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def iter_semantic_tokens(tokens):
    """
  Return a generator over the list of tokens yielding only those that
  have semantic meaning
  """
    for token in tokens:
        if is_semantic_token(token):
            yield token