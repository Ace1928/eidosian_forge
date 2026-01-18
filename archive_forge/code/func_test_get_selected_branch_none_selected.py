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
def test_get_selected_branch_none_selected(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    t = self.get_transport()
    try:
        self.bzrdir_format.initialize(t.base)
    except (errors.NotLocalUrl, errors.UnsupportedOperation):
        raise TestSkipped("Can't initialize %r on transport %r" % (self.bzrdir_format, t))
    dir = controldir.ControlDir.open(t.base)
    self.assertEqual('', dir._get_selected_branch())