import string
from taskflow import test
from taskflow.utils import iter_utils
def test_find_first_match(self):
    it = forever_it()
    self.assertEqual(100, iter_utils.find_first_match(it, lambda v: v == 100))