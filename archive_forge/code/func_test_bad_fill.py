import string
from taskflow import test
from taskflow.utils import iter_utils
def test_bad_fill(self):
    self.assertRaises(ValueError, iter_utils.fill, 2, 2)