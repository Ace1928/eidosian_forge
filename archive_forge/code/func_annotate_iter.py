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
def annotate_iter(self, path, default_revision=_mod_revision.CURRENT_REVISION):
    trans_id = self._path2trans_id(path)
    if trans_id is None:
        return None
    orig_path = self._transform.tree_path(trans_id)
    if orig_path is not None:
        old_annotation = self._transform._tree.annotate_iter(orig_path, default_revision=default_revision)
    else:
        old_annotation = []
    try:
        lines = self.get_file_lines(path)
    except _mod_transport.NoSuchFile:
        return None
    return annotate.reannotate([old_annotation], lines, default_revision)