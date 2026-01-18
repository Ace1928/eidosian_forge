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
def iterobjects_subset(self, shas: Iterable[ObjectID], *, allow_missing: bool=False) -> Iterator[ShaFile]:
    return (uo for uo in PackInflater.for_pack_subset(self, shas, allow_missing=allow_missing, resolve_ext_ref=self.resolve_ext_ref) if uo.id in shas)