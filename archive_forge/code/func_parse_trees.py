import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def parse_trees(cursor_offset, line):
    for i in range(cursor_offset - 1, -1, -1):
        try:
            tree = ast.parse(line[i:cursor_offset])
            yield tree
        except SyntaxError:
            continue