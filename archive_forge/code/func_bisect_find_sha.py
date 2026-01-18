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
def bisect_find_sha(start, end, sha, unpack_name):
    """Find a SHA in a data blob with sorted SHAs.

    Args:
      start: Start index of range to search
      end: End index of range to search
      sha: Sha to find
      unpack_name: Callback to retrieve SHA by index
    Returns: Index of the SHA, or None if it wasn't found
    """
    assert start <= end
    while start <= end:
        i = (start + end) // 2
        file_sha = unpack_name(i)
        if file_sha < sha:
            start = i + 1
        elif file_sha > sha:
            end = i - 1
        else:
            return i
    return None