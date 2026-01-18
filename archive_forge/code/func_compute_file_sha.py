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
def compute_file_sha(f, start_ofs=0, end_ofs=0, buffer_size=1 << 16):
    """Hash a portion of a file into a new SHA.

    Args:
      f: A file-like object to read from that supports seek().
      start_ofs: The offset in the file to start reading at.
      end_ofs: The offset in the file to end reading at, relative to the
        end of the file.
      buffer_size: A buffer size for reading.
    Returns: A new SHA object updated with data read from the file.
    """
    sha = sha1()
    f.seek(0, SEEK_END)
    length = f.tell()
    if end_ofs < 0 and length + end_ofs < start_ofs or end_ofs > length:
        raise AssertionError('Attempt to read beyond file length. start_ofs: %d, end_ofs: %d, file length: %d' % (start_ofs, end_ofs, length))
    todo = length + end_ofs - start_ofs
    f.seek(start_ofs)
    while todo:
        data = f.read(min(todo, buffer_size))
        sha.update(data)
        todo -= len(data)
    return sha