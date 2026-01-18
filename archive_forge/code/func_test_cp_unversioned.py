import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_cp_unversioned(self):
    self.build_tree(['unversioned.txt'])
    self.run_bzr_error(['^brz: ERROR: Could not copy .*unversioned.txt => .*elsewhere. .*unversioned.txt is not versioned\\.$'], 'cp unversioned.txt elsewhere')