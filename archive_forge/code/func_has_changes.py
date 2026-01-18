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
def has_changes(self, _from_tree=None):
    """Quickly check that the tree contains at least one commitable change.

        :param _from_tree: tree to compare against to find changes (default to
            the basis tree and is intended to be used by tests).

        :return: True if a change is found. False otherwise
        """
    with self.lock_read():
        if len(self.get_parent_ids()) > 1:
            return True
        if _from_tree is None:
            _from_tree = self.basis_tree()
        changes = self.iter_changes(_from_tree)
        if self.supports_symlinks():
            try:
                change = next(changes)
                if change.path[1] == '':
                    next(changes)
                return True
            except StopIteration:
                return False
        else:
            changes = filter(lambda c: c[6][0] != 'symlink' and c[4] != (None, None), changes)
            try:
                next(iter(changes))
            except StopIteration:
                return False
            return True