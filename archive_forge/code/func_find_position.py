from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def find_position(string, index, last_index, last_pos):
    """Given a string and index, return (line, column)"""
    lines = string.count('\n', last_index, index)
    if lines > 0:
        column = index - string.rfind('\n', last_index, index)
    else:
        column = last_pos[1] + (index - last_index)
    return (last_pos[0] + lines, column)