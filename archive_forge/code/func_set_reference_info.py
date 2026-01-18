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
def set_reference_info(self, tree_path, branch_location):
    path = self.abspath('.gitmodules')
    try:
        config = GitConfigFile.from_path(path)
    except OSError as e:
        if e.errno == errno.ENOENT:
            config = GitConfigFile()
        else:
            raise
    section = (b'submodule', encode_git_path(tree_path))
    if branch_location is None:
        try:
            del config[section]
        except KeyError:
            pass
    else:
        branch_location = urlutils.join(urlutils.strip_segment_parameters(self.branch.user_url), branch_location)
        config.set(section, b'path', encode_git_path(tree_path))
        config.set(section, b'url', branch_location.encode('utf-8'))
    config.write_to_path(path)
    self.add('.gitmodules')