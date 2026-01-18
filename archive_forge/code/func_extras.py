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
def extras(self):
    possible_extras = {self._transform.trans_id_tree_path(p) for p in self._transform._tree.extras()}
    possible_extras.update(self._transform._new_contents)
    possible_extras.update(self._transform._removed_id)
    for trans_id in possible_extras:
        if not self._transform.final_is_versioned(trans_id):
            yield self._final_paths._determine_path(trans_id)