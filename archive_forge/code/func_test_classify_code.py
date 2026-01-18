from ...tests import TestCase
from .classify import classify_filename
def test_classify_code(self):
    self.assertEqual('code', classify_filename('foo/bar.c'))
    self.assertEqual('code', classify_filename('foo/bar.pl'))
    self.assertEqual('code', classify_filename('foo/bar.pm'))