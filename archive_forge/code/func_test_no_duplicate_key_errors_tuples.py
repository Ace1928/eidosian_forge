from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_tuples(self):
    self.flakes('\n        {(0,1): 1, (0,2): 1}\n        ')