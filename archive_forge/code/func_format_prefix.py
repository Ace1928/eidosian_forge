from __future__ import print_function, unicode_literals
import sys
import typing
from fs.path import abspath, join, normpath
def format_prefix(prefix):
    """Format the prefix lines."""
    if not with_color:
        return prefix
    return '\x1b[32m%s\x1b[0m' % prefix