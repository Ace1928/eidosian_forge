import os
import sys
import stat
import fnmatch
import collections
import errno
def get_unpack_formats():
    """Returns a list of supported formats for unpacking.

    Each element of the returned sequence is a tuple
    (name, extensions, description)
    """
    formats = [(name, info[0], info[3]) for name, info in _UNPACK_FORMATS.items()]
    formats.sort()
    return formats