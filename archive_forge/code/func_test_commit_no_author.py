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
def test_commit_no_author(self):
    """The default kwarg author in MutableTree.commit should not add
        the 'author' revision property.
        """
    tree = self.make_branch_and_tree('foo')
    rev_id = tree.commit('commit 1')
    rev = tree.branch.repository.get_revision(rev_id)
    self.assertFalse('author' in rev.properties)
    self.assertFalse('authors' in rev.properties)