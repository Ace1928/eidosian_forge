from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def only_comments_and_whitespace_remain(tokens, breakstack):
    skip_tokens = (lex.TokenType.WHITESPACE, lex.TokenType.NEWLINE, lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT)
    for token in tokens:
        if token.type in skip_tokens:
            continue
        if should_break(token, breakstack):
            return True
        return False
    return True