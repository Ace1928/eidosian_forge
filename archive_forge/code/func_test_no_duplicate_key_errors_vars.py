from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_no_duplicate_key_errors_vars(self):
    self.flakes("\n        test = 'yes'\n        rest = 'yes'\n        {test: 1, rest: 2}\n        ")