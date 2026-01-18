from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_keys_bytes_vs_unicode_py3(self):
    self.flakes("{b'a': 1, u'a': 2}")