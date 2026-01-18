import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_apply_delta_to_source_is_safe(self):
    self.assertRaises(TypeError, self.apply_delta_to_source, object(), 0, 1)
    self.assertRaises(TypeError, self.apply_delta_to_source, 'unicode str', 0, 1)
    self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 1, 4)
    self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 5, 3)
    self.assertRaises(ValueError, self.apply_delta_to_source, b'foo', 3, 2)