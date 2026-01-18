import errno
import os
import stat
import sys
from subprocess import check_output
def ensure_unicode_string(value):
    """
    Return the given ``value`` as unicode string.

    If the given ``value`` is not a unicode string, but a byte string, it is
    decoded with the filesystem encoding (as in
    :func:`sys.getfilesystemencoding()`).
    """
    if not isinstance(value, str):
        value = value.decode(sys.getfilesystemencoding())
    return value