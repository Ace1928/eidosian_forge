import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_ls(self):
    txt = self.run_bzr_decode('ls')
    self.assertEqual(sorted(['a', 'b', self.info['filename']]), sorted(txt.splitlines()))
    txt = self.run_bzr_decode('ls --null')
    self.assertEqual(sorted(['', 'a', 'b', self.info['filename']]), sorted(txt.split('\x00')))
    txt = self.run_bzr_decode('ls', encoding='ascii', fail=True)
    txt = self.run_bzr_decode('ls --null', encoding='ascii', fail=True)