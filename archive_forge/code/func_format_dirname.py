from __future__ import print_function, unicode_literals
import sys
import typing
from fs.path import abspath, join, normpath
def format_dirname(dirname):
    """Format a directory name."""
    if not with_color:
        return dirname
    return '\x1b[1;34m%s\x1b[0m' % dirname