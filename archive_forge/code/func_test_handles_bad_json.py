from taskflow import test
from taskflow.utils import misc
def test_handles_bad_json(self):
    self.assertRaises(ValueError, misc.decode_json, _bytes('{"foo":'))