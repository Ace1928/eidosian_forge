from taskflow import test
from taskflow.utils import misc
def test_handles_wrong_types(self):
    self.assertRaises(ValueError, misc.decode_json, _bytes('42'))