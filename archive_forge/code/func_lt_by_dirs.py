import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def lt_by_dirs(path1, path2):
    """Compare two paths directory by directory.

    This is equivalent to doing::

       operator.lt(path1.split('/'), path2.split('/'))

    The idea is that you should compare path components separately. This
    differs from plain ``path1 < path2`` for paths like ``'a-b'`` and ``a/b``.
    "a-b" comes after "a" but would come before "a/b" lexically.

    :param path1: first path
    :param path2: second path
    :return: True if path1 comes first, otherwise False
    """
    if not isinstance(path1, bytes):
        raise TypeError("'path1' must be a byte string, not %s: %r" % (type(path1), path1))
    if not isinstance(path2, bytes):
        raise TypeError("'path2' must be a byte string, not %s: %r" % (type(path2), path2))
    return path1.split(b'/') < path2.split(b'/')