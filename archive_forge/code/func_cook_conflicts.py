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
def cook_conflicts(self, raw_conflicts):
    """Generate a list of cooked conflicts, sorted by file path"""
    if not raw_conflicts:
        return
    fp = FinalPaths(self)
    from .workingtree import TextConflict, ContentsConflict
    for c in raw_conflicts:
        if c[0] == 'text conflict':
            yield TextConflict(fp.get_path(c[1]))
        elif c[0] == 'contents conflict':
            yield ContentsConflict(fp.get_path(c[1][0]))
        elif c[0] == 'duplicate':
            yield TextConflict(fp.get_path(c[2]))
        elif c[0] == 'missing parent':
            pass
        elif c[0] == 'non-directory parent':
            yield TextConflict(fp.get_path(c[2]))
        elif c[0] == 'deleting parent':
            yield TextConflict(fp.get_path(c[2]))
        elif c[0] == 'parent loop':
            yield TextConflict(fp.get_path(c[2]))
        else:
            raise AssertionError('unknown conflict %s' % c[0])