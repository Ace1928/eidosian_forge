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
def blob_from_path_and_stat(fs_path: bytes, st, tree_encoding='utf-8'):
    """Create a blob from a path and a stat object.

    Args:
      fs_path: Full file system path to file
      st: A stat object
    Returns: A `Blob` object
    """
    return blob_from_path_and_mode(fs_path, st.st_mode, tree_encoding)