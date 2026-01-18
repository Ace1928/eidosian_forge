import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
def test_incremental(self):
    self.make_git_repo('d')
    os.chdir('d')
    bb = GitBranchBuilder()
    bb.set_file('foobar', b'foo\nbar\n', False)
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
    bb.set_file('foobar', b'fooll\nbar\n', False)
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'nextmsg')
    marks = bb.finish()
    gitsha1 = marks[mark1]
    gitsha2 = marks[mark2]
    os.chdir('..')
    oldrepo = self.open_git_repo('d')
    revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
    newrepo = self.clone_git_repo('d', 'f', revision_id=revid1)
    self.assertEqual([revid1], newrepo.all_revision_ids())
    revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    newrepo.fetch(oldrepo, revision_id=revid2)
    self.assertEqual({revid1, revid2}, set(newrepo.all_revision_ids()))