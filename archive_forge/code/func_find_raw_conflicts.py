import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
def find_raw_conflicts(self):
    """Find any violations of inventory or filesystem invariants"""
    if self._done is True:
        raise ReusingTransform()
    conflicts = []
    self._add_tree_children()
    by_parent = self.by_parent()
    conflicts.extend(self._parent_loops())
    conflicts.extend(self._duplicate_entries(by_parent))
    conflicts.extend(self._parent_type_conflicts(by_parent))
    conflicts.extend(self._improper_versioning())
    conflicts.extend(self._executability_conflicts())
    conflicts.extend(self._overwrite_conflicts())
    return conflicts