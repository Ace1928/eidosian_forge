import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def assertEncode(self, expected, offset, length):
    data = _groupcompress_py.encode_copy_instruction(offset, length)
    self.assertEqual(expected, data)