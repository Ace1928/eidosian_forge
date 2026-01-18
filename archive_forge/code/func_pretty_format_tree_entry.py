import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def pretty_format_tree_entry(name, mode, hexsha, encoding='utf-8') -> str:
    """Pretty format tree entry.

    Args:
      name: Name of the directory entry
      mode: Mode of entry
      hexsha: Hexsha of the referenced object
    Returns: string describing the tree entry
    """
    if mode & stat.S_IFDIR:
        kind = 'tree'
    else:
        kind = 'blob'
    return '{:04o} {} {}\t{}\n'.format(mode, kind, hexsha.decode('ascii'), name.decode(encoding, 'replace'))