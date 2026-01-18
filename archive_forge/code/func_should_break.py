from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def should_break(token, breakstack):
    """
  Return true if any function in breakstack evaluates to true on the current
  token. Otherwise return false.
  """
    for breakcheck in breakstack[::-1]:
        if breakcheck(token):
            return True
    return False