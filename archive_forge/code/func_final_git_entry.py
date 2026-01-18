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
def final_git_entry(self, trans_id):
    if trans_id in self._new_contents:
        path = self._limbo_name(trans_id)
        st = os.lstat(path)
        kind = mode_kind(st.st_mode)
        if kind == 'directory':
            return (None, None)
        executable = mode_is_executable(st.st_mode)
        mode = object_mode(kind, executable)
        blob = blob_from_path_and_stat(encode_git_path(path), st)
    elif trans_id in self._removed_contents:
        return (None, None)
    else:
        orig_path = self.tree_path(trans_id)
        kind = self._tree.kind(orig_path)
        executable = self._tree.is_executable(orig_path)
        mode = object_mode(kind, executable)
        if kind == 'symlink':
            contents = self._tree.get_symlink_target(orig_path)
        elif kind == 'file':
            contents = self._tree.get_file_text(orig_path)
        elif kind == 'directory':
            return (None, None)
        else:
            raise AssertionError(kind)
        blob = Blob.from_string(contents)
    return (blob, mode)