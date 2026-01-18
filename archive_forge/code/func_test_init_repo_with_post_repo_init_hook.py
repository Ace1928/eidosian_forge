import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_init_repo_with_post_repo_init_hook(self):
    calls = []
    ControlDir.hooks.install_named_hook('post_repo_init', calls.append, None)
    self.assertLength(0, calls)
    self.run_bzr('init-shared-repository a')
    self.assertLength(1, calls)