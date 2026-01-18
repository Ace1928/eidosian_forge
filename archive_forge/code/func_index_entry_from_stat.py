import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def index_entry_from_stat(stat_val, hex_sha: bytes, mode: Optional[int]=None):
    """Create a new index entry from a stat value.

    Args:
      stat_val: POSIX stat_result instance
      hex_sha: Hex sha of the object
    """
    if mode is None:
        mode = cleanup_mode(stat_val.st_mode)
    return IndexEntry(stat_val.st_ctime, stat_val.st_mtime, stat_val.st_dev, stat_val.st_ino, mode, stat_val.st_uid, stat_val.st_gid, stat_val.st_size, hex_sha)