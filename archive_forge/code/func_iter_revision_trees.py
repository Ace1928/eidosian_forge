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
def iter_revision_trees(self, revids):
    trees = {}
    todo = []
    for revid in revids:
        try:
            tree = self._cache[revid]
        except KeyError:
            todo.append(revid)
        else:
            if tree.get_revision_id() != revid:
                raise AssertionError('revision id did not match: {} != {}'.format(tree.get_revision_id(), revid))
            trees[revid] = tree
    for tree in self.repository.revision_trees(todo):
        trees[tree.get_revision_id()] = tree
        self.add(tree)
    return (trees[r] for r in revids)