from taskflow import test
from taskflow.utils import misc
def test_handles_invalid_unicode(self):
    self.assertRaises(ValueError, misc.decode_json, '{"ñ": 1}'.encode('latin-1'))