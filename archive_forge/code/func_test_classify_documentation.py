from ...tests import TestCase
from .classify import classify_filename
def test_classify_documentation(self):
    self.assertEqual('documentation', classify_filename('bla.html'))