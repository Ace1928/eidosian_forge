import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def find_last_indent(lines):
    m = _indent_re.match(lines[-1])
    if not m:
        return 0
    return len(m.group(0).replace('\t', ' ' * 4))