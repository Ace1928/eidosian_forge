from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def isolate_expression(string, start_pos, end_pos):
    srow, scol = start_pos
    srow -= 1
    erow, ecol = end_pos
    erow -= 1
    lines = string.splitlines(True)
    if srow == erow:
        return lines[srow][scol:ecol]
    parts = [lines[srow][scol:]]
    parts.extend(lines[srow + 1:erow])
    if erow < len(lines):
        parts.append(lines[erow][:ecol])
    return ''.join(parts)