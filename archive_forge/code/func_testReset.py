import os
import shutil
import stat
import sys
from ...controldir import ControlDir
from .. import KnownFailure, TestCaseWithTransport, TestSkipped
def testReset(self):
    """Test resetting the tree."""
    self.run_bzr(['bisect', 'start'])
    self.run_bzr(['bisect', 'yes'])
    self.run_bzr(['bisect', 'no', '-r', '1'])
    self.run_bzr(['bisect', 'yes'])
    self.run_bzr(['bisect', 'reset'])
    self.assertRevno(5)
    with open('test_file', 'w') as test_file:
        test_file.write('keep me')
    out, err = self.run_bzr(['bisect', 'reset'], retcode=3)
    self.assertIn('No bisection in progress.', err)
    with open('test_file') as test_file:
        content = test_file.read().strip()
    self.assertEqual(content, 'keep me')