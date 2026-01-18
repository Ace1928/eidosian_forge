import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
def load_pack_index_file(path, f):
    """Load an index file from a file-like object.

    Args:
      path: Path for the index file
      f: File-like object
    Returns: A PackIndex loaded from the given file
    """
    contents, size = _load_file_contents(f)
    if contents[:4] == b'\xfftOc':
        version = struct.unpack(b'>L', contents[4:8])[0]
        if version == 2:
            return PackIndex2(path, file=f, contents=contents, size=size)
        else:
            raise KeyError('Unknown pack index format %d' % version)
    else:
        return PackIndex1(path, file=f, contents=contents, size=size)