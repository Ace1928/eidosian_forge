from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def iteratepath(path):
    """Iterate over the individual components of a path.

    Arguments:
        path (str): Path to iterate over.

    Returns:
        list: A list of path components.

    Example:
        >>> iteratepath('/foo/bar/baz')
        ['foo', 'bar', 'baz']

    """
    path = relpath(normpath(path))
    if not path:
        return []
    return path.split('/')