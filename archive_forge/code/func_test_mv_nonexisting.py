import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_nonexisting(self):
    self.run_bzr_error(['^brz: ERROR: Could not rename doesnotexist => somewhereelse. .*doesnotexist is not versioned\\.$'], 'mv doesnotexist somewhereelse')