from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def expand_indent(line):
    """Return the amount of indentation.

    Tabs are expanded to the next multiple of 8.

    >>> expand_indent('    ')
    4
    >>> expand_indent('\\t')
    8
    >>> expand_indent('       \\t')
    8
    >>> expand_indent('        \\t')
    16
    """
    if '\t' not in line:
        return len(line) - len(line.lstrip())
    result = 0
    for char in line:
        if char == '\t':
            result = result // 8 * 8 + 8
        elif char == ' ':
            result += 1
        else:
            break
    return result