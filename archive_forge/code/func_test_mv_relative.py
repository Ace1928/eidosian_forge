import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_relative(self):
    self.build_tree(['sub1/', 'sub1/sub2/', 'sub1/hello.txt'])
    tree = self.make_branch_and_tree('.')
    tree.add(['sub1', 'sub1/sub2', 'sub1/hello.txt'])
    self.run_bzr('mv ../hello.txt .', working_dir='sub1/sub2')
    self.assertPathExists('sub1/sub2/hello.txt')
    self.run_bzr('mv sub2/hello.txt .', working_dir='sub1')
    self.assertMoved('sub1/sub2/hello.txt', 'sub1/hello.txt')