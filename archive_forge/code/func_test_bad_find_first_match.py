import string
from taskflow import test
from taskflow.utils import iter_utils
def test_bad_find_first_match(self):
    self.assertRaises(ValueError, iter_utils.find_first_match, 2, lambda v: False)