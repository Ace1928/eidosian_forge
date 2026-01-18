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
def serialize_tree(items):
    """Serialize the items in a tree to a text.

    Args:
      items: Sorted iterable over (name, mode, sha) tuples
    Returns: Serialized tree text as chunks
    """
    for name, mode, hexsha in items:
        yield (('%04o' % mode).encode('ascii') + b' ' + name + b'\x00' + hex_to_sha(hexsha))