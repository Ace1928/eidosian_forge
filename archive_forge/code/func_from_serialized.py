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
@classmethod
def from_serialized(cls, serialized: SerializedIndexEntry) -> 'IndexEntry':
    return cls(ctime=serialized.ctime, mtime=serialized.mtime, dev=serialized.dev, ino=serialized.ino, mode=serialized.mode, uid=serialized.uid, gid=serialized.gid, size=serialized.size, sha=serialized.sha)