import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def make_tokens_by_line(lines: List[str]):
    """Tokenize a series of lines and group tokens by line.

    The tokens for a multiline Python string or expression are grouped as one
    line. All lines except the last lines should keep their line ending ('\\n',
    '\\r\\n') for this to properly work. Use `.splitlines(keeplineending=True)`
    for example when passing block of text to this function.

    """
    NEWLINE, NL = (tokenize.NEWLINE, tokenize.NL)
    tokens_by_line: List[List[Any]] = [[]]
    if len(lines) > 1 and (not lines[0].endswith(('\n', '\r', '\r\n', '\x0b', '\x0c'))):
        warnings.warn("`make_tokens_by_line` received a list of lines which do not have lineending markers ('\\n', '\\r', '\\r\\n', '\\x0b', '\\x0c'), behavior will be unspecified", stacklevel=2)
    parenlev = 0
    try:
        for token in tokenutil.generate_tokens_catch_errors(iter(lines).__next__, extra_errors_to_catch=['expected EOF']):
            tokens_by_line[-1].append(token)
            if token.type == NEWLINE or (token.type == NL and parenlev <= 0):
                tokens_by_line.append([])
            elif token.string in {'(', '[', '{'}:
                parenlev += 1
            elif token.string in {')', ']', '}'}:
                if parenlev > 0:
                    parenlev -= 1
    except tokenize.TokenError:
        pass
    if not tokens_by_line[-1]:
        tokens_by_line.pop()
    return tokens_by_line