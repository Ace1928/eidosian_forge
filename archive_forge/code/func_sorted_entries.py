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
def sorted_entries(self, progress: Optional[ProgressFn]=None) -> Iterator[PackIndexEntry]:
    """Return entries in this pack, sorted by SHA.

        Args:
          progress: Progress function, called with current and total
            object count
        Returns: Iterator of tuples with (sha, offset, crc32)
        """
    return self.data.sorted_entries(progress=progress, resolve_ext_ref=self.resolve_ext_ref)