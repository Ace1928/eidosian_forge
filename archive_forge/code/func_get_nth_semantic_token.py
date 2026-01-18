from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def get_nth_semantic_token(tokens, nth):
    idx = 0
    for token in iter_semantic_tokens(tokens):
        if idx == nth:
            return token
        idx += 1
    return None