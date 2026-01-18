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
def new_paths(self, filesystem_only=False):
    """Determine the paths of all new and changed files.

        :param filesystem_only: if True, only calculate values for files
            that require renames or execute bit changes.
        """
    new_ids = set()
    if filesystem_only:
        stale_ids = self._needs_rename.difference(self._new_name)
        stale_ids.difference_update(self._new_parent)
        stale_ids.difference_update(self._new_contents)
        stale_ids.difference_update(self._versioned)
        needs_rename = self._needs_rename.difference(stale_ids)
        id_sets = (needs_rename, self._new_executability)
    else:
        id_sets = (self._new_name, self._new_parent, self._new_contents, self._versioned, self._new_executability)
    for id_set in id_sets:
        new_ids.update(id_set)
    return sorted(FinalPaths(self).get_paths(new_ids))