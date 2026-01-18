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
def test_get_branch_reference_on_non_reference(self):
    """get_branch_reference should return None for non-reference branches."""
    dir = self.make_controldir('referenced')
    dir.create_repository()
    if dir._format.colocated_branches:
        name = 'foo'
    else:
        name = None
    branch = dir.create_branch(name)
    self.assertEqual(None, branch.controldir.get_branch_reference(name))