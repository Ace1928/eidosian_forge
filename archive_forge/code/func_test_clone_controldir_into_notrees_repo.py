import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_clone_controldir_into_notrees_repo(self):
    """Cloning into a no-trees repo should not create a working tree"""
    tree = self.make_branch_and_tree('source')
    self.build_tree(['source/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    try:
        repo = self.make_repository('repo', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('must support shared repositories')
    if repo.make_working_trees():
        repo.set_make_working_trees(False)
        self.assertFalse(repo.make_working_trees())
    a_dir = tree.controldir.clone(self.get_url('repo/a'))
    a_branch = a_dir.open_branch()
    if not a_branch.repository.has_same_location(repo):
        raise TestNotApplicable('new control dir does not use repository')
    self.assertRaises(errors.NoWorkingTree, a_dir.open_workingtree)