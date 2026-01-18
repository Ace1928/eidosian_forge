from __future__ import print_function, unicode_literals
import typing
import re
from .errors import IllegalBackReference
def recursepath(path, reverse=False):
    """Get intermediate paths from the root to the given path.

    Arguments:
        path (str): A PyFilesystem path
        reverse (bool): Reverses the order of the paths
            (default `False`).

    Returns:
        list: A list of paths.

    Example:
        >>> recursepath('a/b/c')
        ['/', '/a', '/a/b', '/a/b/c']

    """
    if path in '/':
        return ['/']
    path = abspath(normpath(path)) + '/'
    paths = ['/']
    find = path.find
    append = paths.append
    pos = 1
    len_path = len(path)
    while pos < len_path:
        pos = find('/', pos)
        append(path[:pos])
        pos += 1
    if reverse:
        return paths[::-1]
    return paths