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
def add_objects(self, objects: Sequence[Tuple[ShaFile, Optional[str]]], progress: Optional[Callable[[str], None]]=None) -> None:
    """Add a set of objects to this object store.

        Args:
          objects: Iterable over a list of (object, path) tuples
        """