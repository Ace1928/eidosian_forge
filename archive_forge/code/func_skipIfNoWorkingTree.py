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
def skipIfNoWorkingTree(self, a_controldir):
    """Raises TestSkipped if a_controldir doesn't have a working tree.

        If the bzrdir does have a workingtree, this is a no-op.
        """
    try:
        a_controldir.open_workingtree()
    except (errors.NotLocalUrl, errors.NoWorkingTree):
        raise TestSkipped('bzrdir on transport %r has no working tree' % a_controldir.transport)