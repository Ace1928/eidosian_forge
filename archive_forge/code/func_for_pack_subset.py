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
@classmethod
def for_pack_subset(cls, pack: 'Pack', shas: Iterable[bytes], *, allow_missing: bool=False, resolve_ext_ref=None):
    walker = cls(None, resolve_ext_ref=resolve_ext_ref)
    walker.set_pack_data(pack.data)
    todo = set()
    for sha in shas:
        assert isinstance(sha, bytes)
        try:
            off = pack.index.object_offset(sha)
        except KeyError:
            if not allow_missing:
                raise
        else:
            todo.add(off)
    done = set()
    while todo:
        off = todo.pop()
        unpacked = pack.data.get_unpacked_object_at(off)
        walker.record(unpacked)
        done.add(off)
        base_ofs = None
        if unpacked.pack_type_num == OFS_DELTA:
            base_ofs = unpacked.offset - unpacked.delta_base
        elif unpacked.pack_type_num == REF_DELTA:
            with suppress(KeyError):
                assert isinstance(unpacked.delta_base, bytes)
                base_ofs = pack.index.object_index(unpacked.delta_base)
        if base_ofs is not None and base_ofs not in done:
            todo.add(base_ofs)
    return walker