from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_multiple_duplicate_keys(self):
    self.flakes("{'yes': 1, 'yes': 2, 'no': 2, 'no': 3}", m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)