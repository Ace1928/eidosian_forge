import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_invalid(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['test.txt', 'sub1/'])
    tree.add(['test.txt'])
    self.run_bzr_error(['^brz: ERROR: Could not copy test.txt => sub1/test.txt: sub1 is not versioned\\.$'], 'cp test.txt sub1')
    self.run_bzr_error(['^brz: ERROR: Could not copy test.txt => .*hello.txt: sub1 is not versioned\\.$'], 'cp test.txt sub1/hello.txt')