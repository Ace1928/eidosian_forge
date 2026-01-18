import string
from taskflow import test
from taskflow.utils import iter_utils
def test_bad_unique_seen(self):
    iters = [['a', 'b'], 2, None, object()]
    self.assertRaises(ValueError, iter_utils.unique_seen, iters)