from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def isbase(path1, path2):
    """Check if ``path1`` is a base of ``path2``.

    Arguments:
        path1 (str): A PyFilesytem path.
        path2 (str): A PyFilesytem path.

    Returns:
        bool: `True` if ``path2`` starts with ``path1``

    Example:
        >>> isbase('foo/bar', 'foo/bar/baz/egg.txt')
        True

    """
    _path1 = forcedir(abspath(path1))
    _path2 = forcedir(abspath(path2))
    return _path2.startswith(_path1)