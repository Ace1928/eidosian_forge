import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_commit_ordering(self):
    """Test of corner-case commit ordering error"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/z/', 'a/c/', 'a/z/x', 'a/z/y'])
    tree.add(['a/', 'a/z/', 'a/c/', 'a/z/x', 'a/z/y'])
    tree.commit('setup')
    self.build_tree(['a/c/d/'])
    tree.add('a/c/d')
    tree.rename_one('a/z/x', 'a/c/d/x')
    tree.commit('test', specific_files=['a/z/y'])