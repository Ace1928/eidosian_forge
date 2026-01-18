import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_file_id(self):
    fname = self.info['filename']
    txt = self.run_bzr_decode(['file-id', fname])
    txt = self.run_bzr_decode(['file-id', fname], encoding='ascii')