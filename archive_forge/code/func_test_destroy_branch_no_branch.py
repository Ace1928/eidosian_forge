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
def test_destroy_branch_no_branch(self):
    branch = self.make_repository('branch')
    bzrdir = branch.controldir
    try:
        self.assertRaises(errors.NotBranchError, bzrdir.destroy_branch)
    except (errors.UnsupportedOperation, errors.TransportNotPossible):
        raise TestNotApplicable('Format does not support destroying branch')