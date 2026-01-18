from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def are_column_aligned(token_a, token_b):
    """
  Return true if both tokens are on the same column.
  """
    return token_a.begin.col == token_b.begin.col