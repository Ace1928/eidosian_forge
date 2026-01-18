from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def isparent(path1, path2):
    """Check if ``path1`` is a parent directory of ``path2``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if ``path1`` is a parent directory of ``path2``

    Example:
        >>> isparent("foo/bar", "foo/bar/spam.txt")
        True
        >>> isparent("foo/bar/", "foo/bar")
        True
        >>> isparent("foo/barry", "foo/baz/bar")
        False
        >>> isparent("foo/bar/baz/", "foo/baz/bar")
        False

    """
    bits1 = path1.split('/')
    bits2 = path2.split('/')
    while bits1 and bits1[-1] == '':
        bits1.pop()
    if len(bits1) > len(bits2):
        return False
    for bit1, bit2 in zip(bits1, bits2):
        if bit1 != bit2:
            return False
    return True