from ...tests import TestCase
from .classify import classify_filename
def test_classify_translation(self):
    self.assertEqual('translation', classify_filename('nl.po'))