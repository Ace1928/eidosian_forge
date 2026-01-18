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
def test_interbranch_pull_with_tags(self):
    path, (gitsha1, gitsha2) = self.make_tworev_branch()
    gitrepo = GitRepo(path)
    self.addCleanup(gitrepo.close)
    gitrepo.refs[b'refs/tags/sometag'] = gitsha2
    oldrepo = Repository.open(path)
    revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
    revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    newbranch = self.make_branch('g')
    source_branch = Branch.open(path)
    source_branch.get_config().set_user_option('branch.fetch_tags', True)
    inter_branch = InterBranch.get(source_branch, newbranch)
    inter_branch.pull(stop_revision=revid1)
    self.assertEqual(revid1, newbranch.last_revision())
    self.assertTrue(newbranch.repository.has_revision(revid2))