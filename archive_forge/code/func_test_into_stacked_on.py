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
def test_into_stacked_on(self):
    r = self.make_git_repo('d')
    os.chdir('d')
    bb = GitBranchBuilder()
    bb.set_file('foobar', b'foo\n', False)
    mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
    gitsha1 = bb.finish()[mark1]
    os.chdir('..')
    stacked_on = self.clone_git_repo('d', 'stacked-on')
    oldrepo = Repository.open('d')
    revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
    self.assertEqual([revid1], stacked_on.all_revision_ids())
    b = stacked_on.controldir.create_branch()
    b.generate_revision_history(revid1)
    self.assertEqual(b.last_revision(), revid1)
    tree = self.make_branch_and_tree('stacked')
    tree.branch.set_stacked_on_url(b.user_url)
    os.chdir('d')
    bb = GitBranchBuilder()
    bb.set_file('barbar', b'bar\n', False)
    bb.set_file('foo/blie/bla', b'bla\n', False)
    mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
    gitsha2 = bb.finish()[mark2]
    revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
    os.chdir('..')
    tree.branch.fetch(Branch.open('d'))
    tree.branch.repository.check()
    self.addCleanup(tree.lock_read().unlock)
    self.assertEqual({(revid2,)}, tree.branch.repository.revisions.without_fallbacks().keys())
    self.assertEqual({revid1, revid2}, set(tree.branch.repository.all_revision_ids()))