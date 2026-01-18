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
def generate_lossy_pack_data(self, have, want, shallow=None, progress=None, get_tagged=None, ofs_delta=False):
    object_ids = list(self.find_missing_objects(have, want, progress=progress, shallow=shallow, get_tagged=get_tagged, lossy=True))
    return pack_objects_to_data([(self[oid], path) for oid, (type_num, path) in object_ids])