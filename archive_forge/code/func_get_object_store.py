import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def get_object_store(repo, mapping=None):
    git = getattr(repo, '_git', None)
    if git is not None:
        git.object_store.unlock = lambda: None
        git.object_store.lock_read = lambda: LogicalLockResult(lambda: None)
        git.object_store.lock_write = lambda: LogicalLockResult(lambda: None)
        return git.object_store
    return BazaarObjectStore(repo, mapping)