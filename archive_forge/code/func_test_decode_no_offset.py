import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_decode_no_offset(self):
    self.assertDecode(0, 1, 2, b'\x90\x01', 0)
    self.assertDecode(0, 10, 2, b'\x90\n', 0)
    self.assertDecode(0, 255, 2, b'\x90\xff', 0)
    self.assertDecode(0, 256, 2, b'\xa0\x01', 0)
    self.assertDecode(0, 257, 3, b'\xb0\x01\x01', 0)
    self.assertDecode(0, 65535, 3, b'\xb0\xff\xff', 0)
    self.assertDecode(0, 65536, 1, b'\x80', 0)