import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
def test_last_revision_is_null(self):
    r = GitRepo.init('.')
    thedir = ControlDir.open('.')
    thebranch = thedir.create_branch()
    self.assertEqual(revision.NULL_REVISION, thebranch.last_revision())
    self.assertEqual((0, revision.NULL_REVISION), thebranch.last_revision_info())