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
def iter_fresh_entries(paths: Iterable[bytes], root_path: bytes, object_store: Optional[ObjectContainer]=None) -> Iterator[Tuple[bytes, Optional[IndexEntry]]]:
    """Iterate over current versions of index entries on disk.

    Args:
      paths: Paths to iterate over
      root_path: Root path to access from
      object_store: Optional store to save new blobs in
    Returns: Iterator over path, index_entry
    """
    for path in paths:
        p = _tree_to_fs_path(root_path, path)
        try:
            entry = index_entry_from_path(p, object_store=object_store)
        except (FileNotFoundError, IsADirectoryError):
            entry = None
        yield (path, entry)