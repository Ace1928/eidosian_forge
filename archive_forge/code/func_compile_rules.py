import re
import typing as t
from ast import literal_eval
from collections import deque
from sys import intern
from ._identifier import pattern as name_re
from .exceptions import TemplateSyntaxError
from .utils import LRUCache
def compile_rules(environment: 'Environment') -> t.List[t.Tuple[str, str]]:
    """Compiles all the rules from the environment into a list of rules."""
    e = re.escape
    rules = [(len(environment.comment_start_string), TOKEN_COMMENT_BEGIN, e(environment.comment_start_string)), (len(environment.block_start_string), TOKEN_BLOCK_BEGIN, e(environment.block_start_string)), (len(environment.variable_start_string), TOKEN_VARIABLE_BEGIN, e(environment.variable_start_string))]
    if environment.line_statement_prefix is not None:
        rules.append((len(environment.line_statement_prefix), TOKEN_LINESTATEMENT_BEGIN, '^[ \\t\\v]*' + e(environment.line_statement_prefix)))
    if environment.line_comment_prefix is not None:
        rules.append((len(environment.line_comment_prefix), TOKEN_LINECOMMENT_BEGIN, '(?:^|(?<=\\S))[^\\S\\r\\n]*' + e(environment.line_comment_prefix)))
    return [x[1:] for x in sorted(rules, reverse=True)]