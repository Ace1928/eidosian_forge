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
def iter_child_entries(self, path):
    trans_id = self._path2trans_id(path)
    if trans_id is None:
        raise _mod_transport.NoSuchFile(path)
    for child_trans_id in self._all_children(trans_id):
        entry, is_versioned = self._transform.final_entry(trans_id)
        if not is_versioned:
            continue
        if entry is not None:
            yield entry