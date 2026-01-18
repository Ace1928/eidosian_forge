from ...tests import TestCase
from .classify import classify_filename
def test_classify_unknown(self):
    self.assertEqual(None, classify_filename('something.bar'))