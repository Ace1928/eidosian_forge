from ...tests import TestCase
from .classify import classify_filename
def test_classify_art(self):
    self.assertEqual('art', classify_filename('icon.png'))