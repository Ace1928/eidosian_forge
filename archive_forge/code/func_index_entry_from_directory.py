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
def index_entry_from_directory(st, path: bytes) -> Optional[IndexEntry]:
    if os.path.exists(os.path.join(path, b'.git')):
        head = read_submodule_head(path)
        if head is None:
            return None
        return index_entry_from_stat(st, head, mode=S_IFGITLINK)
    return None