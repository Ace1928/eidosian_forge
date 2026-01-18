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
def simple_commit_a(self):
    r = GitRepo.init('.')
    self.build_tree(['a'])
    r.stage(['a'])
    return r.do_commit(b'a', committer=b'Somebody <foo@example.com>')