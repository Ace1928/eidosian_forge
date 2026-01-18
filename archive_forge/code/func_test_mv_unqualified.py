import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_unqualified(self):
    self.run_bzr_error(['^brz: ERROR: missing file argument$'], 'mv')