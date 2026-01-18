import string
from taskflow import test
from taskflow.utils import iter_utils
def test_fill_many_empty(self):
    result = list(iter_utils.fill(range(0, 50), 500))
    self.assertEqual(450, sum((1 for x in result if x is None)))
    self.assertEqual(50, sum((1 for x in result if x is not None)))