import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def process_signature(line: str) -> str:
    """
    Clean up a given raw function signature.

    This includes removing the self-referential datapipe argument, default
    arguments of input functions, newlines, and spaces.
    """
    tokens: List[str] = split_outside_bracket(line)
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(' ')
        if token == 'cls':
            tokens[i] = 'self'
        elif i > 0 and 'self' == tokens[i - 1] and (tokens[i][0] != '*'):
            tokens[i] = ''
        elif 'Callable =' in token:
            head, default_arg = token.rsplit('=', 2)
            tokens[i] = head.strip(' ') + '= ...'
    tokens = [t for t in tokens if t != '']
    line = ', '.join(tokens)
    return line