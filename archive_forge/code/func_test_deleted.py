import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_deleted(self):
    self._check_OSX_can_roundtrip(self.info['filename'])
    fname = self.info['filename']
    os.remove(fname)
    self.wt.remove(fname)
    txt = self.run_bzr_decode('deleted')
    self.assertEqual(fname + '\n', txt)
    txt = self.run_bzr_decode('deleted --show-ids')
    self.assertTrue(txt.startswith(fname))
    self.run_bzr_decode('deleted', encoding='ascii', fail=True)