import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_smoke_aliases(self):
    self.build_tree(['a'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a'])
    self.run_bzr('move a b')
    self.run_bzr('rename b a')