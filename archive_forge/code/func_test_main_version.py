import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def test_main_version(self):
    """Check output from version command and master option is reasonable"""
    self.permit_source_tree_branch_repo()
    output = self.run_bzr('version')[0]
    self.log('brz version output:')
    self.log(output)
    self.assertTrue(output.startswith('Breezy (brz) '))
    self.assertNotEqual(output.index('Canonical'), -1)
    tmp_output = self.run_bzr('--version')[0]
    self.assertEqual(output, tmp_output)