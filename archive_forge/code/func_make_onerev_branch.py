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
def make_onerev_branch(self):
    os.mkdir('d')
    os.chdir('d')
    GitRepo.init('.')
    bb = tests.GitBranchBuilder()
    bb.set_file('foobar', b'foo\nbar\n', False)
    mark = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    gitsha = bb.finish()[mark]
    os.chdir('..')
    return (os.path.abspath('d'), gitsha)