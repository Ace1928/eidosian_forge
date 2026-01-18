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
def create_index_v1(self, filename, progress=None, resolve_ext_ref=None):
    """Create a version 1 file for this data file.

        Args:
          filename: Index filename.
          progress: Progress report function
        Returns: Checksum of index file
        """
    entries = self.sorted_entries(progress=progress, resolve_ext_ref=resolve_ext_ref)
    with GitFile(filename, 'wb') as f:
        return write_pack_index_v1(f, entries, self.calculate_checksum())