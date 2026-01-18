import os
import sys
import stat
import fnmatch
import collections
import errno
def get_archive_formats():
    """Returns a list of supported formats for archiving and unarchiving.

    Each element of the returned sequence is a tuple (name, description)
    """
    formats = [(name, registry[2]) for name, registry in _ARCHIVE_FORMATS.items()]
    formats.sort()
    return formats