from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def iswildcard(path):
    """Check if a path ends with a wildcard.

    Arguments:
        path (str): A PyFilesystem path.

    Returns:
        bool: `True` if path ends with a wildcard.

    Example:
        >>> iswildcard('foo/bar/baz.*')
        True
        >>> iswildcard('foo/bar')
        False

    """
    assert path is not None
    return not _WILD_CHARS.isdisjoint(path)