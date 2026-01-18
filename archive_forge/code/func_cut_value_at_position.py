import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def cut_value_at_position(leaf, position):
    """
    Cuts of the value of the leaf at position
    """
    lines = split_lines(leaf.value, keepends=True)[:position[0] - leaf.line + 1]
    column = position[1]
    if leaf.line == position[0]:
        column -= leaf.column
    if not lines:
        return ''
    lines[-1] = lines[-1][:column]
    return ''.join(lines)