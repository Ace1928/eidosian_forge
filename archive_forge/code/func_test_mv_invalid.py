import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_invalid(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['test.txt', 'sub1/'])
    tree.add(['test.txt'])
    self.run_bzr_error(['^brz: ERROR: Could not move to sub1: sub1 is not versioned\\.$'], 'mv test.txt sub1')
    self.run_bzr_error(['^brz: ERROR: Could not move test.txt => .*hello.txt: sub1 is not versioned\\.$'], 'mv test.txt sub1/hello.txt')