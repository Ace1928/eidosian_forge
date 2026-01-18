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
def test_clone_controldir_with_colocated(self):
    if not self.bzrdir_format.colocated_branches:
        raise TestNotApplicable('format does not supported colocated branches')
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    rev1 = tree.commit('revision 1')
    rev2 = tree.commit('revision 2', allow_pointless=True)
    rev3 = tree.commit('revision 2', allow_pointless=True)
    dir = tree.branch.controldir
    colo = dir.create_branch(name='colo')
    colo.pull(tree.branch, stop_revision=rev1)
    target = dir.clone(self.get_url('target'), revision_id=rev2)
    self.assertEqual(rev2, target.open_branch().last_revision())
    self.assertEqual(rev1, target.open_branch(name='colo').last_revision())