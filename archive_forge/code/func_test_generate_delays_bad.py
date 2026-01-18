import string
from taskflow import test
from taskflow.utils import iter_utils
def test_generate_delays_bad(self):
    self.assertRaises(ValueError, iter_utils.generate_delays, -1, -1)
    self.assertRaises(ValueError, iter_utils.generate_delays, -1, 2)
    self.assertRaises(ValueError, iter_utils.generate_delays, 2, -1)
    self.assertRaises(ValueError, iter_utils.generate_delays, 1, 1, multiplier=0.5)