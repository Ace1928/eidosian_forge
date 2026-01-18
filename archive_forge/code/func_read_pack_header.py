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
def read_pack_header(read) -> Tuple[int, int]:
    """Read the header of a pack file.

    Args:
      read: Read function
    Returns: Tuple of (pack version, number of objects). If no data is
        available to read, returns (None, None).
    """
    header = read(12)
    if not header:
        raise AssertionError('file too short to contain pack')
    if header[:4] != b'PACK':
        raise AssertionError('Invalid pack header %r' % header)
    version, = unpack_from(b'>L', header, 4)
    if version not in (2, 3):
        raise AssertionError('Version was %d' % version)
    num_objects, = unpack_from(b'>L', header, 8)
    return (version, num_objects)