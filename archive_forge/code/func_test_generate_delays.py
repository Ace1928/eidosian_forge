import string
from taskflow import test
from taskflow.utils import iter_utils
def test_generate_delays(self):
    it = iter_utils.generate_delays(1, 60)
    self.assertEqual(1, next(it))
    self.assertEqual(2, next(it))
    self.assertEqual(4, next(it))
    self.assertEqual(8, next(it))
    self.assertEqual(16, next(it))
    self.assertEqual(32, next(it))
    self.assertEqual(60, next(it))
    self.assertEqual(60, next(it))