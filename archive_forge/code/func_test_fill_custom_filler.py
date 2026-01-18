import string
from taskflow import test
from taskflow.utils import iter_utils
def test_fill_custom_filler(self):
    self.assertEqual('abcd', ''.join(iter_utils.fill('abc', 4, filler='d')))