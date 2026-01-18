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
def get_unpacked_object_at(self, offset: int, *, include_comp: bool=False) -> UnpackedObject:
    """Given offset in the packfile return a UnpackedObject."""
    assert offset >= self._header_size
    self._file.seek(offset)
    unpacked, _ = unpack_object(self._file.read, include_comp=include_comp)
    unpacked.offset = offset
    return unpacked