from collections import namedtuple
from io import StringIO
from keyword import iskeyword
import tokenize
from tokenize import TokenInfo
from typing import List, Optional
def generate_tokens_catch_errors(readline, extra_errors_to_catch: Optional[List[str]]=None):
    default_errors_to_catch = ['unterminated string literal', 'invalid non-printable character', 'after line continuation character']
    assert extra_errors_to_catch is None or isinstance(extra_errors_to_catch, list)
    errors_to_catch = default_errors_to_catch + (extra_errors_to_catch or [])
    tokens: List[TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(readline):
            tokens.append(token)
            yield token
    except tokenize.TokenError as exc:
        if any((error in exc.args[0] for error in errors_to_catch)):
            if tokens:
                start = (tokens[-1].start[0], tokens[-1].end[0])
                end = start
                line = tokens[-1].line
            else:
                start = end = (1, 0)
                line = ''
            yield tokenize.TokenInfo(tokenize.ERRORTOKEN, '', start, end, line)
        else:
            raise