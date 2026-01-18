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
def find_related_paths_across_trees(self, paths, trees=[], require_versioned=True):
    if paths is None:
        return None

    def include(t, p):
        if t.is_versioned(p):
            return True
        try:
            if t.kind(p) == 'directory':
                return True
        except _mod_transport.NoSuchFile:
            return False
        return False
    if require_versioned:
        trees = [self] + (trees if trees is not None else [])
        unversioned = set()
        for p in paths:
            for t in trees:
                if include(t, p):
                    break
            else:
                unversioned.add(p)
        if unversioned:
            raise errors.PathsNotVersionedError(unversioned)
    return filter(partial(include, self), paths)