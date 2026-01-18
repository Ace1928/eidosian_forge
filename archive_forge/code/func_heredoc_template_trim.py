import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def heredoc_template_trim(self, args: List) -> str:
    match = HEREDOC_TRIM_PATTERN.match(str(args[0]))
    if not match:
        raise RuntimeError(f'Invalid Heredoc token: {args[0]}')
    trim_chars = '\n\t '
    text = match.group(2).rstrip(trim_chars)
    lines = text.split('\n')
    min_spaces = sys.maxsize
    for line in lines:
        leading_spaces = len(line) - len(line.lstrip(' '))
        min_spaces = min(min_spaces, leading_spaces)
    lines = [line[min_spaces:] for line in lines]
    return '"%s"' % '\n'.join(lines)