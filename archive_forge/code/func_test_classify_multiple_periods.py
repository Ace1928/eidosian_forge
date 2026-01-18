from ...tests import TestCase
from .classify import classify_filename
def test_classify_multiple_periods(self):
    self.assertEqual('documentation', classify_filename('foo.bla.html'))