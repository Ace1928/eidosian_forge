import errno
import itertools
import os
import posixpath
import re
import stat
import sys
from collections import defaultdict
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.file import FileLocked, GitFile
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (ConflictedIndexEntry, Index, IndexEntry, SHA1Writer,
from dulwich.object_store import iter_tree_contents
from dulwich.objects import S_ISGITLINK
from .. import branch as _mod_branch
from .. import conflicts as _mod_conflicts
from .. import controldir as _mod_controldir
from .. import errors, globbing, lock, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, urlutils, workingtree
from ..decorators import only_raises
from ..mutabletree import BadReferenceTarget, MutableTree
from .dir import BareLocalGitControlDirFormat, LocalGitDir
from .mapping import decode_git_path, encode_git_path, mode_kind
from .tree import MutableGitIndexTree
def get_parent_ids(self):
    """See Tree.get_parent_ids.

        This implementation reads the pending merges list and last_revision
        value and uses that to decide what the parents list should be.
        """
    last_rev = self._last_revision()
    if _mod_revision.NULL_REVISION == last_rev:
        parents = []
    else:
        parents = [last_rev]
    try:
        merges_bytes = self.control_transport.get_bytes('MERGE_HEAD')
    except _mod_transport.NoSuchFile:
        pass
    else:
        for l in osutils.split_lines(merges_bytes):
            revision_id = l.rstrip(b'\n')
            parents.append(self.branch.lookup_foreign_revision_id(revision_id))
    return parents