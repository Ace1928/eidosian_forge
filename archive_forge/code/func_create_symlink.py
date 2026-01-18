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
def create_symlink(self, target, trans_id):
    """Schedule creation of a new symbolic link.

        target is a bytestring.
        See also new_symlink.
        """
    if self._create_symlinks:
        os.symlink(target, self._limbo_name(trans_id))
    else:
        try:
            path = FinalPaths(self).get_path(trans_id)
        except KeyError:
            path = None
        trace.warning('Unable to create symlink "{}" on this filesystem.'.format(path))
        self._symlink_target[trans_id] = target
    unique_add(self._new_contents, trans_id, 'symlink')