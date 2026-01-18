import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def get_lexer(environment: 'Environment') -> 'Lexer':
    """Return a lexer which is probably cached."""
    key = (environment.block_start_string, environment.block_end_string, environment.variable_start_string, environment.variable_end_string, environment.comment_start_string, environment.comment_end_string, environment.line_statement_prefix, environment.line_comment_prefix, environment.trim_blocks, environment.lstrip_blocks, environment.newline_sequence, environment.keep_trailing_newline)
    lexer = _lexer_cache.get(key)
    if lexer is None:
        _lexer_cache[key] = lexer = Lexer(environment)
    return lexer