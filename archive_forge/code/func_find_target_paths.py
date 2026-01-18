import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
def find_target_paths(self, paths, recurse='none'):
    paths = set(paths)
    ret = {}
    changes = self._iter_git_changes(specific_files=paths, include_trees=False)[0]
    for change_type, old, new in changes:
        if old[0] is None:
            continue
        oldpath = decode_git_path(old[0])
        if oldpath in paths:
            ret[oldpath] = decode_git_path(new[0]) if new[0] else None
    for path in paths:
        if path not in ret:
            if self.source.has_filename(path):
                if self.target.has_filename(path):
                    ret[path] = path
                else:
                    ret[path] = None
            else:
                raise _mod_transport.NoSuchFile(path)
    return ret