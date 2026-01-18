from .. import cache_utf8
from . import TestCase
def test_ascii(self):
    self.check_decode('all_ascii_characters123123123')
    self.check_encode('all_ascii_characters123123123')