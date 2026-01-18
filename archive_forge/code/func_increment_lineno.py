import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def increment_lineno(node, n=1):
    """
    Increment the line number and end line number of each node in the tree
    starting at *node* by *n*. This is useful to "move code" to a different
    location in a file.
    """
    for child in walk(node):
        if isinstance(child, TypeIgnore):
            child.lineno = getattr(child, 'lineno', 0) + n
            continue
        if 'lineno' in child._attributes:
            child.lineno = getattr(child, 'lineno', 0) + n
        if 'end_lineno' in child._attributes and (end_lineno := getattr(child, 'end_lineno', 0)) is not None:
            child.end_lineno = end_lineno + n
    return node