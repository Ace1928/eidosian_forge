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
def test_find_repository_containing_shared_repository(self):
    try:
        repo = self.make_repository('.', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('requires format with shared repository support')
    if not repo._format.supports_nesting_repositories:
        raise TestNotApplicable('requires support for nesting repositories')
    url = self.get_url('childbzrdir')
    self.get_transport().mkdir('childbzrdir')
    made_control = self.bzrdir_format.initialize(url)
    try:
        made_control.open_repository()
        return
    except errors.NoRepositoryPresent:
        pass
    found_repo = made_control.find_repository()
    self.assertEqual(repo.controldir.root_transport.base, found_repo.controldir.root_transport.base)