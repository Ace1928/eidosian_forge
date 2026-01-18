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
def test_clone_bzrdir_branch_reference(self):
    referenced_branch = self.make_branch('referencced')
    dir = self.make_controldir('source')
    try:
        dir.set_branch_reference(referenced_branch)
    except errors.IncompatibleFormat:
        return
    target = dir.clone(self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport)