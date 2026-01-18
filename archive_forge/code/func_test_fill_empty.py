import string
from taskflow import test
from taskflow.utils import iter_utils
def test_fill_empty(self):
    self.assertEqual([], list(iter_utils.fill([1, 2, 3], 0)))