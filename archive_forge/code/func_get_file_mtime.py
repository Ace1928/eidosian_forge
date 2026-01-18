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
def get_file_mtime(self, path):
    """See Tree.get_file_mtime"""
    trans_id = self._path2trans_id(path)
    if trans_id is None:
        raise _mod_transport.NoSuchFile(path)
    if trans_id not in self._transform._new_contents:
        return self._transform._tree.get_file_mtime(self._transform.tree_path(trans_id))
    name = self._transform._limbo_name(trans_id)
    statval = os.lstat(name)
    return statval.st_mtime