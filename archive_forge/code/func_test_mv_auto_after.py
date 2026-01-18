import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_auto_after(self):
    self.make_abcd_tree()
    out, err = self.run_bzr('mv --auto --after', working_dir='tree', retcode=3)
    self.assertEqual('brz: ERROR: --after cannot be specified with --auto.\n', err)