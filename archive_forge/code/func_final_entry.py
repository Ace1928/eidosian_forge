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
def final_entry(self, trans_id):
    is_versioned = self.final_is_versioned(trans_id)
    fp = FinalPaths(self)
    tree_path = fp.get_path(trans_id)
    if trans_id in self._new_contents:
        path = self._limbo_name(trans_id)
        st = os.lstat(path)
        kind = mode_kind(st.st_mode)
        name = self.final_name(trans_id)
        file_id = self._tree.mapping.generate_file_id(tree_path)
        parent_id = self._tree.mapping.generate_file_id(os.path.dirname(tree_path))
        if kind == 'directory':
            return (GitTreeDirectory(file_id, self.final_name(trans_id), parent_id=parent_id), is_versioned)
        executable = mode_is_executable(st.st_mode)
        mode = object_mode(kind, executable)
        blob = blob_from_path_and_stat(encode_git_path(path), st)
        if kind == 'symlink':
            return (GitTreeSymlink(file_id, name, parent_id, decode_git_path(blob.data)), is_versioned)
        elif kind == 'file':
            return (GitTreeFile(file_id, name, executable=executable, parent_id=parent_id, git_sha1=blob.id, text_size=len(blob.data)), is_versioned)
        else:
            raise AssertionError(kind)
    elif trans_id in self._removed_contents:
        return (None, None)
    else:
        orig_path = self.tree_path(trans_id)
        if orig_path is None:
            return (None, None)
        file_id = self._tree.mapping.generate_file_id(tree_path)
        if tree_path == '':
            parent_id = None
        else:
            parent_id = self._tree.mapping.generate_file_id(os.path.dirname(tree_path))
        try:
            ie = next(self._tree.iter_entries_by_dir(specific_files=[orig_path]))[1]
            ie.file_id = file_id
            ie.parent_id = parent_id
            return (ie, is_versioned)
        except StopIteration:
            try:
                if self.tree_kind(trans_id) == 'directory':
                    return (GitTreeDirectory(file_id, self.final_name(trans_id), parent_id=parent_id), is_versioned)
            except OSError as e:
                if e.errno != errno.ENOTDIR:
                    raise
            return (None, None)