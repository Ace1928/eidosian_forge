from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def partial_tokens(s):
    """Iterate over tokens from a possibly-incomplete string of code.

    This adds two special token types: INCOMPLETE_STRING and
    IN_MULTILINE_STATEMENT. These can only occur as the last token yielded, and
    represent the two main ways for code to be incomplete.
    """
    readline = io.StringIO(s).readline
    token = tokenize.TokenInfo(tokenize.NEWLINE, '', (1, 0), (1, 0), '')
    try:
        for token in tokenutil.generate_tokens_catch_errors(readline):
            yield token
    except tokenize.TokenError as e:
        lines = s.splitlines(keepends=True)
        end = (len(lines), len(lines[-1]))
        if 'multi-line string' in e.args[0]:
            l, c = start = token.end
            s = lines[l - 1][c:] + ''.join(lines[l:])
            yield IncompleteString(s, start, end, lines[-1])
        elif 'multi-line statement' in e.args[0]:
            yield InMultilineStatement(end, lines[-1])
        else:
            raise