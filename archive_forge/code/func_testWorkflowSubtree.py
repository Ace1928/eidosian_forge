import os
import shutil
import stat
import sys
from ...controldir import ControlDir
from .. import KnownFailure, TestCaseWithTransport, TestSkipped
def testWorkflowSubtree(self):
    """Run through a usage scenario where the offending change
        is in a subtree."""
    self.run_bzr(['bisect', 'start'])
    self.run_bzr(['bisect', 'yes'])
    self.run_bzr(['bisect', 'no', '-r', '1'])
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(2)
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(1.2)
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(1.1)
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(1.1)
    self.run_bzr(['bisect', 'yes'])
    self.assertRevno(1.1)