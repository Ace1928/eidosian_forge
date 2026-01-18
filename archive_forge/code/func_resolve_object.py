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
def resolve_object(self, offset: int, type: int, obj, get_ref=None) -> Tuple[int, Iterable[bytes]]:
    """Resolve an object, possibly resolving deltas when necessary.

        Returns: Tuple with object type and contents.
        """
    base_offset = offset
    base_type = type
    base_obj = obj
    delta_stack = []
    while base_type in DELTA_TYPES:
        prev_offset = base_offset
        if get_ref is None:
            get_ref = self.get_ref
        if base_type == OFS_DELTA:
            delta_offset, delta = base_obj
            base_offset = base_offset - delta_offset
            base_type, base_obj = self.data.get_object_at(base_offset)
            assert isinstance(base_type, int)
        elif base_type == REF_DELTA:
            basename, delta = base_obj
            assert isinstance(basename, bytes) and len(basename) == 20
            base_offset, base_type, base_obj = get_ref(basename)
            assert isinstance(base_type, int)
        delta_stack.append((prev_offset, base_type, delta))
    chunks = base_obj
    for prev_offset, delta_type, delta in reversed(delta_stack):
        chunks = apply_delta(chunks, delta)
        if prev_offset is not None:
            self.data._offset_cache[prev_offset] = (base_type, chunks)
    return (base_type, chunks)