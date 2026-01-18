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
def test_format_initialize_on_transport_ex_force_new_repo_False(self):
    t = self.get_transport('repo')
    repo_fmt = controldir.format_registry.make_controldir('1.9')
    repo_name = repo_fmt.repository_format.network_name()
    repo = repo_fmt.initialize_on_transport_ex(t, repo_format_name=repo_name, shared_repo=True)[0]
    made_repo, control = self.assertInitializeEx(t.clone('branch'), force_new_repo=False, repo_format_name=repo_name)
    if not control._format.fixed_components:
        self.assertEqual(repo.controldir.root_transport.base, made_repo.controldir.root_transport.base)