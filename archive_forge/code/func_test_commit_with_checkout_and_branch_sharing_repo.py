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
def test_commit_with_checkout_and_branch_sharing_repo(self):
    repo = self.make_repository('repo', shared=True)
    branch = controldir.ControlDir.create_branch_convenience('repo/branch')
    tree2 = branch.create_checkout('repo/tree2')
    tree2.commit('message', rev_id=b'rev1')
    self.assertTrue(tree2.branch.repository.has_revision(b'rev1'))