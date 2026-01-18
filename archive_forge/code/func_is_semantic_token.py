from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def is_semantic_token(token):
    """Return true for everything that isnt' whitespace or a comment"""
    if token.type in NON_SEMANTIC_TOKENS:
        return False
    return True