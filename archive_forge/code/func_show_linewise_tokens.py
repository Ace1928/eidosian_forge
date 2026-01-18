import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def show_linewise_tokens(s: str):
    """For investigation and debugging"""
    warnings.warn('show_linewise_tokens is deprecated since IPython 8.6', DeprecationWarning, stacklevel=2)
    if not s.endswith('\n'):
        s += '\n'
    lines = s.splitlines(keepends=True)
    for line in make_tokens_by_line(lines):
        print('Line -------')
        for tokinfo in line:
            print(' ', tokinfo)