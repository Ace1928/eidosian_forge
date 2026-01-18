import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_clone_on_transport(self):
    a_dir = self.make_controldir('source')
    target_transport = a_dir.root_transport.clone('..').clone('target')
    target = a_dir.clone_on_transport(target_transport)
    self.assertNotEqual(a_dir.transport.base, target.transport.base)
    self.assertDirectoriesEqual(a_dir.root_transport, target.root_transport, ['./.bzr/merge-hashes'])