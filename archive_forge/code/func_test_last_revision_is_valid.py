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
def test_last_revision_is_valid(self):
    head = self.simple_commit_a()
    thebranch = Branch.open('.')
    self.assertEqual(default_mapping.revision_id_foreign_to_bzr(head), thebranch.last_revision())