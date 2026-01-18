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
@staticmethod
def from_raw_chunks(type_num: int, chunks: List[bytes], sha: Optional[ObjectID]=None):
    """Creates an object of the indicated type from the raw chunks given.

        Args:
          type_num: The numeric type of the object.
          chunks: An iterable of the raw uncompressed contents.
          sha: Optional known sha for the object
        """
    cls = object_class(type_num)
    if cls is None:
        raise AssertionError('unsupported class type num: %d' % type_num)
    obj = cls()
    obj.set_raw_chunks(chunks, sha)
    return obj