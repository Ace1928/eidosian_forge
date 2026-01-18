import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_mkdir_fails_cleanly(self):
    """'mkdir' fails cleanly when no working tree is available.
        https://bugs.launchpad.net/bzr/+bug/138600
        """
    shared_repo = self.make_repository('.')
    self.run_bzr(['mkdir', 'abc'], retcode=3)
    self.assertPathDoesNotExist('abc')