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
def adjust_path(self, name, parent, trans_id):
    previous_parent = self._new_parent.get(trans_id)
    previous_name = self._new_name.get(trans_id)
    super().adjust_path(name, parent, trans_id)
    if trans_id in self._limbo_files and trans_id not in self._needs_rename:
        self._rename_in_limbo([trans_id])
        if previous_parent != parent:
            self._limbo_children[previous_parent].remove(trans_id)
        if previous_parent != parent or previous_name != name:
            del self._limbo_children_names[previous_parent][previous_name]