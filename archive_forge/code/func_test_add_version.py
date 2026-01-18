from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_add_version(self):
    vf = self.make_vf()
    self.assertEqual(REV_A, vf._lines[b'rev-a'])
    vf.clear_cache()
    self.assertEqual(vf._lines, {})