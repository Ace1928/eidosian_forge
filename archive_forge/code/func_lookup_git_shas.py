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
def lookup_git_shas(self, shas: Iterable[ObjectID]) -> Dict[ObjectID, List]:
    ret: Dict[ObjectID, List] = {}
    for sha in shas:
        if sha == ZERO_SHA:
            ret[sha] = [('commit', (NULL_REVISION, None, {}))]
            continue
        try:
            ret[sha] = list(self._cache.idmap.lookup_git_sha(sha))
        except KeyError:
            self._update_sha_map()
            try:
                ret[sha] = list(self._cache.idmap.lookup_git_sha(sha))
            except KeyError:
                pass
    return ret