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
def test_format_initialize_on_transport_ex_repo_fmt_name_followed(self):
    t = self.get_transport('dir')
    fmt = controldir.format_registry.make_controldir('1.6')
    repo_name = fmt.repository_format.network_name()
    repo, control = self.assertInitializeEx(t, repo_format_name=repo_name)
    if self.bzrdir_format.fixed_components:
        repo_name = self.bzrdir_format.network_name()
    self.assertEqual(repo_name, repo._format.network_name())