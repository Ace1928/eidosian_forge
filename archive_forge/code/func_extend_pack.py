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
def extend_pack(f: BinaryIO, object_ids: Set[ObjectID], get_raw, *, compression_level=-1, progress=None) -> Tuple[bytes, List]:
    """Extend a pack file with more objects.

    The caller should make sure that object_ids does not contain any objects
    that are already in the pack
    """
    f.seek(0)
    _version, num_objects = read_pack_header(f.read)
    if object_ids:
        f.seek(0)
        write_pack_header(f.write, num_objects + len(object_ids))
        f.flush()
    new_sha = compute_file_sha(f, end_ofs=-20)
    f.seek(0, os.SEEK_CUR)
    extra_entries = []
    for i, object_id in enumerate(object_ids):
        if progress is not None:
            progress(('writing extra base objects: %d/%d\r' % (i, len(object_ids))).encode('ascii'))
        assert len(object_id) == 20
        type_num, data = get_raw(object_id)
        offset = f.tell()
        crc32 = write_pack_object(f.write, type_num, data, sha=new_sha, compression_level=compression_level)
        extra_entries.append((object_id, offset, crc32))
    pack_sha = new_sha.digest()
    f.write(pack_sha)
    return (pack_sha, extra_entries)