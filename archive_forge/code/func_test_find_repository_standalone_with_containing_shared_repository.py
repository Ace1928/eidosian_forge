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
def test_find_repository_standalone_with_containing_shared_repository(self):
    try:
        containing_repo = self.make_repository('.', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('requires support for shared repositories')
    if not containing_repo._format.supports_nesting_repositories:
        raise TestNotApplicable('format does not support nesting repositories')
    child_repo = self.make_repository('childrepo')
    opened_control = controldir.ControlDir.open(self.get_url('childrepo'))
    found_repo = opened_control.find_repository()
    self.assertEqual(child_repo.controldir.root_transport.base, found_repo.controldir.root_transport.base)