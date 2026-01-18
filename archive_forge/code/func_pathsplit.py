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
def pathsplit(path: bytes) -> Tuple[bytes, bytes]:
    """Split a /-delimited path into a directory part and a basename.

    Args:
      path: The path to split.

    Returns:
      Tuple with directory name and basename
    """
    try:
        dirname, basename = path.rsplit(b'/', 1)
    except ValueError:
        return (b'', path)
    else:
        return (dirname, basename)