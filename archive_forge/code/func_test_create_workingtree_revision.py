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
def test_create_workingtree_revision(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    t = self.get_transport()
    source = self.make_branch_and_tree('source')
    a = source.commit('a', allow_pointless=True)
    b = source.commit('b', allow_pointless=True)
    t.mkdir('new')
    t_new = t.clone('new')
    made_control = self.bzrdir_format.initialize_on_transport(t_new)
    source.branch.repository.clone(made_control)
    source.branch.clone(made_control)
    try:
        made_tree = made_control.create_workingtree(revision_id=a)
    except (errors.NotLocalUrl, errors.UnsupportedOperation):
        raise TestSkipped("Can't make working tree on transport %r" % t)
    self.assertEqual([a], made_tree.get_parent_ids())