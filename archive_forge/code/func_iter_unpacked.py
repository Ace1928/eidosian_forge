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
def iter_unpacked(self, include_comp=False):
    ofs_to_entries = {ofs: (sha, crc32) for sha, ofs, crc32 in self.index.iterentries()}
    for unpacked in self.data.iter_unpacked(include_comp=include_comp):
        sha, crc32 = ofs_to_entries[unpacked.offset]
        unpacked._sha = sha
        unpacked.crc32 = crc32
        yield unpacked