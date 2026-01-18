import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def copy_location(new_node, old_node):
    """
    Copy source location (`lineno`, `col_offset`, `end_lineno`, and `end_col_offset`
    attributes) from *old_node* to *new_node* if possible, and return *new_node*.
    """
    for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
        if attr in old_node._attributes and attr in new_node._attributes:
            value = getattr(old_node, attr, None)
            if value is not None or (hasattr(old_node, attr) and attr.startswith('end_')):
                setattr(new_node, attr, value)
    return new_node