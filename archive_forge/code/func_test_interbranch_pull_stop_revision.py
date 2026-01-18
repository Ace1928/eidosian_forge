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
def test_interbranch_pull_stop_revision(self):
    path, (gitsha1, gitsha2) = self.make_tworev_branch()
    oldrepo = Repository.open(path)
    revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
    newbranch = self.make_branch('g')
    inter_branch = InterBranch.get(Branch.open(path), newbranch)
    inter_branch.pull(stop_revision=revid1)
    self.assertEqual(revid1, newbranch.last_revision())