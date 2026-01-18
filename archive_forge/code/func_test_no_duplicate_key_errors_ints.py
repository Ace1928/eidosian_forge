from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_ints(self):
    self.flakes('\n        {1: 1, 2: 1}\n        ')