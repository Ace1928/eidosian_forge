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
def find_reusable_deltas(container: PackedObjectContainer, object_ids: Set[bytes], *, other_haves: Optional[Set[bytes]]=None, progress=None) -> Iterator[UnpackedObject]:
    if other_haves is None:
        other_haves = set()
    reused = 0
    for i, unpacked in enumerate(container.iter_unpacked_subset(object_ids, allow_missing=True, convert_ofs_delta=True)):
        if progress is not None and i % 1000 == 0:
            progress(('checking for reusable deltas: %d/%d\r' % (i, len(object_ids))).encode('utf-8'))
        if unpacked.pack_type_num == REF_DELTA:
            hexsha = sha_to_hex(unpacked.delta_base)
            if hexsha in object_ids or hexsha in other_haves:
                yield unpacked
                reused += 1
    if progress is not None:
        progress(('found %d deltas to reuse\n' % (reused,)).encode('utf-8'))