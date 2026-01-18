from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def forcedir(path):
    """Ensure the path ends with a trailing forward slash.

    Arguments:
        path (str): A PyFilesytem path.

    Returns:
        str: The path, ending with a slash.

    Example:
        >>> forcedir("foo/bar")
        'foo/bar/'
        >>> forcedir("foo/bar/")
        'foo/bar/'
        >>> forcedir("foo/spam.txt")
        'foo/spam.txt/'

    """
    if not path.endswith('/'):
        return path + '/'
    return path