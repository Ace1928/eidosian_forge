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
def make_tworev_branch(self):
    os.mkdir('d')
    os.chdir('d')
    GitRepo.init('.')
    bb = tests.GitBranchBuilder()
    bb.set_file('foobar', b'foo\nbar\n', False)
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    marks = bb.finish()
    os.chdir('..')
    return ('d', (marks[mark1], marks[mark2]))