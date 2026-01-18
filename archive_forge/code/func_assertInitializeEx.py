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
def assertInitializeEx(self, t, need_meta=False, **kwargs):
    """Execute initialize_on_transport_ex and check it succeeded correctly.

        This involves checking that the disk objects were created, open with
        the same format returned, and had the expected disk format.

        :param t: The transport to initialize on.
        :param **kwargs: Additional arguments to pass to
            initialize_on_transport_ex.
        :return: the resulting repo, control dir tuple.
        """
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('control dir format is not initializable')
    repo, control, require_stacking, repo_policy = self.bzrdir_format.initialize_on_transport_ex(t, **kwargs)
    if repo is not None:
        self.assertTrue(repo.is_write_locked())
        self.addCleanup(repo.unlock)
    self.assertIsInstance(control, bzrdir.BzrDir)
    opened = bzrdir.BzrDir.open(t.base)
    expected_format = self.bzrdir_format
    if need_meta and expected_format.fixed_components:
        expected_format = bzrdir.BzrDirMetaFormat1()
    if not isinstance(expected_format, RemoteBzrDirFormat):
        self.assertEqual(control._format.network_name(), expected_format.network_name())
        self.assertEqual(control._format.network_name(), opened._format.network_name())
    self.assertEqual(control.__class__, opened.__class__)
    return (repo, control)