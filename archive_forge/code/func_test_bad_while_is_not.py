import string
from taskflow import test
from taskflow.utils import iter_utils
def test_bad_while_is_not(self):
    self.assertRaises(ValueError, iter_utils.while_is_not, 2, 'a')