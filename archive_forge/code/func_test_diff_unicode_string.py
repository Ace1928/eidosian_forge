import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_diff_unicode_string(self):
    a = ''.join([chr(i) for i in range(4000, 4500, 3)])
    b = ''.join([chr(i) for i in range(4300, 4800, 2)])
    sm = self._PatienceSequenceMatcher(None, a, b)
    mb = sm.get_matching_blocks()
    self.assertEqual(35, len(mb))