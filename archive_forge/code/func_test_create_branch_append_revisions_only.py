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
def test_create_branch_append_revisions_only(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    t = self.get_transport()
    made_control = self.bzrdir_format.initialize(t.base)
    made_control.create_repository()
    try:
        made_branch = made_control.create_branch(append_revisions_only=True)
    except errors.UpgradeRequired:
        raise TestNotApplicable('format does not support append_revisions_only setting')
    self.assertIsInstance(made_branch, breezy.branch.Branch)
    self.assertEqual(True, made_branch.get_append_revisions_only())
    self.assertEqual(made_control, made_branch.controldir)