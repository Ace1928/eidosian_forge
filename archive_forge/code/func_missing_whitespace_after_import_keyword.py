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
def missing_whitespace_after_import_keyword(logical_line):
    """Multiple imports in form from x import (a, b, c) should have space
    between import statement and parenthesised name list.

    Okay: from foo import (bar, baz)
    E275: from foo import(bar, baz)
    E275: from importable.module import(bar, baz)
    """
    line = logical_line
    indicator = ' import('
    if line.startswith('from '):
        found = line.find(indicator)
        if -1 < found:
            pos = found + len(indicator) - 1
            yield (pos, 'E275 missing whitespace after keyword')