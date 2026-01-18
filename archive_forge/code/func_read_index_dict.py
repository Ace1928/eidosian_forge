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
def read_index_dict(f) -> Dict[bytes, Union[IndexEntry, ConflictedIndexEntry]]:
    """Read an index file and return it as a dictionary.
       Dict Key is tuple of path and stage number, as
            path alone is not unique
    Args:
      f: File object to read fromls.
    """
    ret: Dict[bytes, Union[IndexEntry, ConflictedIndexEntry]] = {}
    for entry in read_index(f):
        stage = entry.stage()
        if stage == Stage.NORMAL:
            ret[entry.name] = IndexEntry.from_serialized(entry)
        else:
            existing = ret.setdefault(entry.name, ConflictedIndexEntry())
            if isinstance(existing, IndexEntry):
                raise AssertionError('Non-conflicted entry for %r exists' % entry.name)
            if stage == Stage.MERGE_CONFLICT_ANCESTOR:
                existing.ancestor = IndexEntry.from_serialized(entry)
            elif stage == Stage.MERGE_CONFLICT_THIS:
                existing.this = IndexEntry.from_serialized(entry)
            elif stage == Stage.MERGE_CONFLICT_OTHER:
                existing.other = IndexEntry.from_serialized(entry)
    return ret