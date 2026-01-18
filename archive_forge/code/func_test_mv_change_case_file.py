import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_change_case_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['test.txt'])
    tree.add(['test.txt'])
    self.run_bzr('mv test.txt Test.txt')
    shape = sorted(os.listdir('.'))
    self.assertEqual(['.bzr', 'Test.txt'], shape)
    self.assertInWorkingTree('Test.txt')
    self.assertNotInWorkingTree('test.txt')