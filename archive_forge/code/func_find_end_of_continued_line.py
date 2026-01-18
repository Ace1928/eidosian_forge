import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def find_end_of_continued_line(lines, start_line: int):
    """Find the last line of a line explicitly extended using backslashes.

    Uses 0-indexed line numbers.
    """
    end_line = start_line
    while lines[end_line].endswith('\\\n'):
        end_line += 1
        if end_line >= len(lines):
            break
    return end_line