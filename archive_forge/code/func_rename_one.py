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
def rename_one(self, from_rel, to_rel, after=None):
    from_path = encode_git_path(from_rel)
    to_rel, can_access = osutils.normalized_filename(to_rel)
    if not can_access:
        raise errors.InvalidNormalization(to_rel)
    to_path = encode_git_path(to_rel)
    with self.lock_tree_write():
        if not after:
            after = not self.has_filename(from_rel) and self.has_filename(to_rel) and (not self.is_versioned(to_rel))
        if after:
            if not self.has_filename(to_rel):
                raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(to_rel))
            if self.basis_tree().is_versioned(to_rel):
                raise errors.BzrMoveFailedError(from_rel, to_rel, errors.AlreadyVersionedError(to_rel))
            kind = self.kind(to_rel)
        else:
            try:
                to_kind = self.kind(to_rel)
            except _mod_transport.NoSuchFile:
                exc_type = errors.BzrRenameFailedError
                to_kind = None
            else:
                exc_type = errors.BzrMoveFailedError
            if self.is_versioned(to_rel):
                raise exc_type(from_rel, to_rel, errors.AlreadyVersionedError(to_rel))
            if not self.has_filename(from_rel):
                raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(from_rel))
            kind = self.kind(from_rel)
            if not self.is_versioned(from_rel) and kind != 'directory':
                raise exc_type(from_rel, to_rel, errors.NotVersionedError(from_rel))
            if self.has_filename(to_rel):
                raise errors.RenameFailedFilesExist(from_rel, to_rel, _mod_transport.FileExists(to_rel))
            kind = self.kind(from_rel)
        if not after and kind != 'directory':
            index, from_subpath = self._lookup_index(from_path)
            if from_subpath not in index:
                raise errors.BzrMoveFailedError(from_rel, to_rel, errors.NotVersionedError(path=from_rel))
        if not after:
            try:
                self._rename_one(from_rel, to_rel)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    raise errors.BzrMoveFailedError(from_rel, to_rel, _mod_transport.NoSuchFile(to_rel))
                raise
        if kind != 'directory':
            index, from_index_path = self._lookup_index(from_path)
            try:
                self._index_del_entry(index, from_path)
            except KeyError:
                pass
            self._index_add_entry(to_rel, kind)
        else:
            todo = [(p, i) for p, i in self._recurse_index_entries() if p.startswith(from_path + b'/')]
            for child_path, child_value in todo:
                child_to_index, child_to_index_path = self._lookup_index(posixpath.join(to_path, posixpath.relpath(child_path, from_path)))
                child_to_index[child_to_index_path] = child_value
                self._index_dirty = True
                child_from_index, child_from_index_path = self._lookup_index(child_path)
                self._index_del_entry(child_from_index, child_from_index_path)
        self._versioned_dirs = None
        self.flush()