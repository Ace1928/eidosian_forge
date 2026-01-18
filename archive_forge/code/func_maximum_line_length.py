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
def maximum_line_length(physical_line, max_line_length, multiline, noqa):
    """Limit all lines to a maximum of 79 characters.

    There are still many devices around that are limited to 80 character
    lines; plus, limiting windows to 80 characters makes it possible to have
    several windows side-by-side.  The default wrapping on such devices looks
    ugly.  Therefore, please limit all lines to a maximum of 79 characters.
    For flowing long blocks of text (docstrings or comments), limiting the
    length to 72 characters is recommended.

    Reports error E501.
    """
    line = physical_line.rstrip()
    length = len(line)
    if length > max_line_length and (not noqa):
        chunks = line.split()
        if (len(chunks) == 1 and multiline or (len(chunks) == 2 and chunks[0] == '#')) and len(line) - len(chunks[-1]) < max_line_length - 7:
            return
        if hasattr(line, 'decode'):
            try:
                length = len(line.decode('utf-8'))
            except UnicodeError:
                pass
        if length > max_line_length:
            return (max_line_length, 'E501 line too long (%d > %d characters)' % (length, max_line_length))