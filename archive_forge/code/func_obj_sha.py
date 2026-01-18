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
def obj_sha(type, chunks):
    """Compute the SHA for a numeric type and object chunks."""
    sha = sha1()
    sha.update(object_header(type, chunks_length(chunks)))
    if isinstance(chunks, bytes):
        sha.update(chunks)
    else:
        for chunk in chunks:
            sha.update(chunk)
    return sha.digest()