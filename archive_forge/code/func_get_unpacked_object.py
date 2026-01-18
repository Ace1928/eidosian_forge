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
def get_unpacked_object(self, sha: bytes, *, include_comp: bool=False, convert_ofs_delta: bool=True) -> UnpackedObject:
    """Get the unpacked object for a sha.

        Args:
          sha: SHA of object to fetch
          include_comp: Whether to include compression data in UnpackedObject
        """
    offset = self.index.object_offset(sha)
    unpacked = self.data.get_unpacked_object_at(offset, include_comp=include_comp)
    if unpacked.pack_type_num == OFS_DELTA and convert_ofs_delta:
        assert isinstance(unpacked.delta_base, int)
        unpacked.delta_base = self.index.object_sha1(offset - unpacked.delta_base)
        unpacked.pack_type_num = REF_DELTA
    return unpacked