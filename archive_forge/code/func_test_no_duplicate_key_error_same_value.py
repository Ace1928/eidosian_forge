from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_error_same_value(self):
    self.flakes("\n        {'yes': 1, 'yes': 1}\n        ")