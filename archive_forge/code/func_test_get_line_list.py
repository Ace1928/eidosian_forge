from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_get_line_list(self):
    vf = self.make_vf()
    vf.clear_cache()
    self.assertEqual(REV_A, vf.get_line_list([b'rev-a'])[0])
    self.assertEqual([REV_B, REV_C], vf.get_line_list([b'rev-b', b'rev-c']))