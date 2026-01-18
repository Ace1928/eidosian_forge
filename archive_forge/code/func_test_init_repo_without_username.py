import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_init_repo_without_username(self):
    """Ensure init-shared-repo works if username is not set.
        """
    self.overrideEnv('EMAIL', None)
    self.overrideEnv('BRZ_EMAIL', None)
    out, err = self.run_bzr(['init-shared-repo', 'foo'])
    self.assertEqual(err, '')
    self.assertTrue(os.path.exists('foo'))