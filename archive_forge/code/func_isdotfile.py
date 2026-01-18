from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def isdotfile(path):
    """Detect if a path references a dot file.

    Arguments:
        path (str): Path to check.

    Returns:
        bool: `True` if the resource name starts with a ``'.'``.

    Example:
        >>> isdotfile('.baz')
        True
        >>> isdotfile('foo/bar/.baz')
        True
        >>> isdotfile('foo/bar.baz')
        False

    """
    return basename(path).startswith('.')