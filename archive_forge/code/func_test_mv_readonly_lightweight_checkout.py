import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_readonly_lightweight_checkout(self):
    branch = self.make_branch('foo')
    branch = breezy.branch.Branch.open(self.get_readonly_url('foo'))
    tree = branch.create_checkout('tree', lightweight=True)
    self.build_tree(['tree/path'])
    tree.add('path')
    self.run_bzr(['mv', 'tree/path', 'tree/path2'])